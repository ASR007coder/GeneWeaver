
import numpy as np
import pytest
from numba import cuda

from backend.alignment.gpu_alignment import compare_dna_gpu


# Check that CUDA is available before running GPU tests.
@pytest.fixture(scope="module", autouse=True)
def check_cuda():
    if not cuda.is_available():
        pytest.skip("CUDA GPU is not available")


def test_normal_comparison():
    result = compare_dna_gpu("ATGCAGGC", "ATGC")

    expected = np.array([0, 4, 4, 3, 1], dtype=np.int32)

    np.testing.assert_array_equal(result, expected)


def test_target_longer_than_genome():
    result = compare_dna_gpu("AT", "ATGC")

    assert result.size == 0


def test_target_equal_to_genome():
    result = compare_dna_gpu("ATGC", "ATGC")

    expected = np.array([0], dtype=np.int32)

    np.testing.assert_array_equal(result, expected)


def test_single_base_comparison():
    result = compare_dna_gpu("ATGC", "A")

    expected = np.array([0, 1, 1, 1], dtype=np.int32)

    np.testing.assert_array_equal(result, expected)


def test_no_matching_bases():
    result = compare_dna_gpu("AAAA", "TT")

    expected = np.array([2, 2, 2], dtype=np.int32)

    np.testing.assert_array_equal(result, expected)


def test_repeated_sequence():
    result = compare_dna_gpu("AAAAAA", "AAA")

    expected = np.array([0, 0, 0, 0], dtype=np.int32)

    np.testing.assert_array_equal(result, expected)


def test_invalid_dna():
    with pytest.raises(ValueError, match="Invalid DNA base"):
        compare_dna_gpu("ATGN", "AT")


def test_empty_target():
    with pytest.raises(ValueError, match="cannot be empty"):
        compare_dna_gpu("ATGC", "")
