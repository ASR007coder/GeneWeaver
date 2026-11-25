
import numpy as np
from numba import cuda

from backend.alignment.dna_encoding import encode_dna
from backend.alignment.gpu_alignment import compare_dna_kernel


def process_genomic_chunk(
    chunk,
    target_sequence,
    max_mismatches=0
):
    """
    Process one genomic chunk using GPU alignment.

    Expected chunk format:
    {
        "chromosome": "chr1",
        "sequence": "ATGCAGGC",
        "start_position": 100000
    }
    """

    if not isinstance(chunk, dict):
        raise TypeError("Chunk must be a dictionary")

    required_fields = {"chromosome", "sequence", "start_position"}

    if not required_fields.issubset(chunk):
        raise ValueError("Chunk is missing required fields")

    chromosome = chunk["chromosome"]
    sequence = chunk["sequence"]
    start_position = chunk["start_position"]

    if not isinstance(chromosome, str) or not chromosome:
        raise ValueError("Invalid chromosome name")

    if not isinstance(start_position, int) or isinstance(start_position, bool):
        raise TypeError("start_position must be an integer")

    if start_position < 0:
        raise ValueError("start_position cannot be negative")

    if not isinstance(max_mismatches, int) or isinstance(max_mismatches, bool):
        raise TypeError("max_mismatches must be an integer")

    if max_mismatches < 0:
        raise ValueError("max_mismatches cannot be negative")

    genome = encode_dna(sequence)
    target = encode_dna(target_sequence)

    if target.size > genome.size:
        return []

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    num_windows = genome.size - target.size + 1

    try:
        d_genome = cuda.to_device(genome)
        d_target = cuda.to_device(target)

        d_mismatch_counts = cuda.device_array(
            num_windows,
            dtype=np.int32
        )

        d_match_flags = cuda.device_array(
            num_windows,
            dtype=np.uint8
        )

        threads_per_block = 128
        blocks_per_grid = (
            num_windows + threads_per_block - 1
        ) // threads_per_block

        compare_dna_kernel[blocks_per_grid, threads_per_block](
            d_genome,
            d_target,
            d_mismatch_counts,
            d_match_flags,
            max_mismatches
        )

        mismatch_counts = d_mismatch_counts.copy_to_host()
        match_flags = d_match_flags.copy_to_host()

        results = []

        for i in range(num_windows):
            if match_flags[i] == 1:
                results.append({
                    "chromosome": chromosome,
                    "position": start_position + i,
                    "sequence": sequence[
                        i:i + len(target_sequence)
                    ].upper(),
                    "mismatches": int(mismatch_counts[i])
                })

        return results

    finally:
        # Release references to temporary device arrays.
        if "d_genome" in locals():
            del d_genome
        if "d_target" in locals():
            del d_target
        if "d_mismatch_counts" in locals():
            del d_mismatch_counts
        if "d_match_flags" in locals():
            del d_match_flags
