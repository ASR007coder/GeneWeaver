
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError

from backend.alignment.dna_encoding import encode_dna


@cuda.jit
def compare_dna_kernel(
    genome,
    target,
    mismatch_counts,
    match_flags,
    max_mismatches
):
    idx = cuda.grid(1)

    if idx < mismatch_counts.size:
        mismatches = 0

        for j in range(target.size):
            if genome[idx + j] != target[j]:
                mismatches += 1

        mismatch_counts[idx] = mismatches

        if mismatches <= max_mismatches:
            match_flags[idx] = 1
        else:
            match_flags[idx] = 0


class GPUAlignmentWorkspace:
    """
    Reuses the encoded target and its GPU allocation
    across multiple chunk-processing calls.
    """

    def __init__(self):
        self._target_host = None
        self._target_device = None

    def get_target(self, target):
        if (
            self._target_host is None
            or not np.array_equal(self._target_host, target)
        ):
            self._target_device = cuda.to_device(target)
            self._target_host = target.copy()

        return self._target_device

    def close(self):
        self._target_device = None
        self._target_host = None


def compare_dna_gpu(
    genome_sequence,
    target_sequence,
    max_mismatches=0,
    return_flags=False,
    workspace=None
):
    # Validate mismatch threshold.
    if (
        not isinstance(max_mismatches, int)
        or isinstance(max_mismatches, bool)
    ):
        raise TypeError("max_mismatches must be an integer")

    if max_mismatches < 0:
        raise ValueError("max_mismatches cannot be negative")

    # Encode DNA using uint8 arrays.
    genome = encode_dna(genome_sequence)
    target = encode_dna(target_sequence)

    if target.size > genome.size:
        empty_counts = np.empty(0, dtype=np.int32)
        empty_flags = np.empty(0, dtype=np.uint8)

        if return_flags:
            return empty_counts, empty_flags

        return empty_counts

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    num_windows = genome.size - target.size + 1

    # Initialize references for reliable cleanup.
    d_genome = None
    d_target = None
    d_mismatch_counts = None
    d_match_flags = None

    try:
        # Transfer the current genome chunk.
        d_genome = cuda.to_device(genome)

        # Reuse the target allocation when a workspace is supplied.
        if workspace is not None:
            d_target = workspace.get_target(target)
        else:
            d_target = cuda.to_device(target)

        # Allocate output arrays.
        d_mismatch_counts = cuda.device_array(
            num_windows, dtype=np.int32
        )

        d_match_flags = cuda.device_array(
            num_windows, dtype=np.uint8
        )

        threads_per_block = 128

        blocks_per_grid = (
            num_windows + threads_per_block - 1
        ) // threads_per_block

        # Execute CUDA kernel.
        compare_dna_kernel[blocks_per_grid, threads_per_block](
            d_genome,
            d_target,
            d_mismatch_counts,
            d_match_flags,
            max_mismatches
        )

        # Copy results back to CPU.
        mismatch_counts = d_mismatch_counts.copy_to_host()
        match_flags = d_match_flags.copy_to_host()

        if return_flags:
            return mismatch_counts, match_flags

        return mismatch_counts

    except (MemoryError, CudaAPIError) as exc:
        raise RuntimeError(
            "GPU memory allocation or CUDA operation failed. "
            "Try processing a smaller genomic chunk."
        ) from exc

    finally:
        # Release temporary device-array references.
        d_genome = None
        d_mismatch_counts = None
        d_match_flags = None
        d_target = None


def find_gpu_matches(
    genome_sequence,
    target_sequence,
    max_mismatches=0,
    chromosome="chr1",
    start_position=0,
    workspace=None
):
    mismatch_counts, match_flags = compare_dna_gpu(
        genome_sequence,
        target_sequence,
        max_mismatches=max_mismatches,
        return_flags=True,
        workspace=workspace
    )

    results = []

    for i in range(len(match_flags)):
        if match_flags[i] == 1:
            results.append({
                "chromosome": chromosome,
                "window_index": i,
                "position": start_position + i,
                "sequence": genome_sequence[
                    i:i + len(target_sequence)
                ].upper(),
                "mismatches": int(mismatch_counts[i])
            })

    return results


if __name__ == "__main__":
    genome = "ATGCAGGC"
    target = "ATGC"

    workspace = GPUAlignmentWorkspace()

    try:
        counts = compare_dna_gpu(
            genome,
            target,
            max_mismatches=1,
            workspace=workspace
        )

        print("Mismatch Counts:", counts)

        matches = find_gpu_matches(
            genome,
            target,
            max_mismatches=1,
            chromosome="chr1",
            start_position=100,
            workspace=workspace
        )

        print("Matching Windows:")

        for match in matches:
            print(match)

    finally:
        workspace.close()
