# GeneWeaver/backend/gpu_alignment.py
import numpy as np
from numba import cuda

from dna_encoding import encode_dna


# One GPU thread compares one genomic window.
@cuda.jit
def compare_dna_kernel(genome, target, mismatch_counts):

    idx = cuda.grid(1)

    # Make sure this thread has a valid window.
    if idx < mismatch_counts.size:

        mismatches = 0

        for j in range(target.size):
            if genome[idx + j] != target[j]:
                mismatches += 1

        mismatch_counts[idx] = mismatches


def compare_dna_gpu(genome_sequence, target_sequence):

    # Convert DNA strings into numerical arrays.
    genome = encode_dna(genome_sequence)
    target = encode_dna(target_sequence)

    if target.size > genome.size:
        return np.empty(0, dtype=np.int32)

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    # Calculate the number of valid windows.
    num_windows = genome.size - target.size + 1

    # Allocate CPU output if there are no valid windows.
    if num_windows == 0:
        return np.empty(0, dtype=np.int32)

    # Transfer input arrays to GPU.
    d_genome = cuda.to_device(genome)
    d_target = cuda.to_device(target)

    # Allocate GPU output array.
    d_mismatch_counts = cuda.device_array(
        num_windows,
        dtype=np.int32
    )

    # Configure GPU execution.
    threads_per_block = 128
    blocks_per_grid = (
        num_windows + threads_per_block - 1
    ) // threads_per_block

    # Execute CUDA kernel.
    compare_dna_kernel[blocks_per_grid, threads_per_block](
        d_genome,
        d_target,
        d_mismatch_counts
    )

    # Copy results back to CPU.
    mismatch_counts = d_mismatch_counts.copy_to_host()

    return mismatch_counts


if __name__ == "__main__":

    genome = "ATGCAGGC"
    target = "ATGC"

    results = compare_dna_gpu(genome, target)

    print("Original Genome:", genome)
    print("Target DNA:", target)
    print("Mismatch Counts:", results)
