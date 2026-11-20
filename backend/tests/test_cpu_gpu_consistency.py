
import numpy as np
import pytest

from backend.alignment.cpu_alignment import (
    count_mismatches,
    compare_dna
)

from backend.alignment.gpu_alignment import (
    compare_dna_gpu,
    find_gpu_matches
)


@pytest.mark.parametrize(
    "genome, target, max_mismatches, start_position",
    [
        ("ATGCAGGC", "ATGC", 0, 100),       # Exact match
        ("ATGCAGGC", "ATGC", 1, 100),       # Mismatch threshold
        ("TTTTCCCC", "AAAA", 0, 100),       # No matches
        ("ATGATGCCC", "ATG", 0, 100),       # Multiple matches
        ("ATGC", "ATGC", 0, 100),           # Single window
        ("ATG", "ATGC", 0, 100),            # Target longer
    ]
)
def test_cpu_gpu_mismatch_counts_and_matches(
    genome,
    target,
    max_mismatches,
    start_position
):
    # CPU: calculate mismatch count for every possible window
    if len(target) > len(genome):
        cpu_counts = np.array([], dtype=np.int32)
    else:
        cpu_counts = np.array(
            [
                count_mismatches(
                    target,
                    genome[i:i + len(target)]
                )
                for i in range(len(genome) - len(target) + 1)
            ],
            dtype=np.int32
        )

    # GPU: calculate mismatch count for every possible window
    gpu_counts = compare_dna_gpu(genome, target)

    # Compare all mismatch counts
    np.testing.assert_array_equal(cpu_counts, gpu_counts)

    # CPU: identify matching positions
    cpu_matches = compare_dna(
        target,
        genome,
        chromosome="chr1",
        start_position=start_position,
        max_mismatches=max_mismatches
    )

    # GPU: identify matching positions
    gpu_matches = find_gpu_matches(
        genome,
        target,
        max_mismatches=max_mismatches,
        chromosome="chr1",
        start_position=start_position
    )

    # Normalize results for comparison
    cpu_results = [
        (m["position"], m["sequence"], m["mismatches"])
        for m in cpu_matches
    ]

    gpu_results = [
        (m["position"], m["sequence"], m["mismatches"])
        for m in gpu_matches
    ]

    assert cpu_results == gpu_results
