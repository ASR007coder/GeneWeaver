
import numpy as np
import pytest
from numba import cuda

from backend.alignment.gpu_alignment import (
    compare_dna_gpu,
    find_gpu_matches
)


@pytest.fixture(scope="module", autouse=True)
def check_cuda():
    if not cuda.is_available():
        pytest.skip("CUDA GPU is not available")


def test_mismatch_counts():
    result = compare_dna_gpu("ATGCAGGT", "ATGC")

    expected = np.array([0, 4, 4, 3, 2], dtype=np.int32)

    np.testing.assert_array_equal(result, expected)


def test_exact_match_threshold():
    result = compare_dna_gpu(
        "ATGCAGGT", "ATGC",
        max_mismatches=0,
        return_flags=True
    )

    counts, flags = result

    np.testing.assert_array_equal(
        flags, np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    )


def test_one_mismatch_threshold():
    counts, flags = compare_dna_gpu(
        "ATGCAGGT", "ATGC",
        max_mismatches=1,
        return_flags=True
    )

    np.testing.assert_array_equal(
        flags, np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    )


def test_two_mismatch_threshold():
    counts, flags = compare_dna_gpu(
        "ATGCAGGT", "ATGC",
        max_mismatches=2,
        return_flags=True
    )

    np.testing.assert_array_equal(
        flags, np.array([1, 0, 0, 0, 1], dtype=np.uint8)
    )


def test_matching_positions():
    results = find_gpu_matches(
        "ATGCAGGT",
        "ATGC",
        max_mismatches=2,
        chromosome="chr1",
        start_position=100
    )

    positions = [item["position"] for item in results]

    assert positions == [100, 104]


def test_target_longer_than_genome():
    result = compare_dna_gpu("AT", "ATGC")

    assert result.size == 0


def test_negative_threshold():
    with pytest.raises(ValueError):
        compare_dna_gpu("ATGC", "ATGC", max_mismatches=-1)
