
import numpy as np
import pytest

from backend.dna_encoding import encode_dna


def test_encode_dna():
    result = encode_dna("ATGC")

    expected = np.array([0, 1, 2, 3], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_lowercase_sequence():
    result = encode_dna("atgc")

    expected = np.array([0, 1, 2, 3], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_output_data_type():
    result = encode_dna("ATGC")

    assert result.dtype == np.uint8


def test_empty_sequence():
    with pytest.raises(ValueError):
        encode_dna("")


def test_invalid_base():
    with pytest.raises(ValueError, match="Invalid DNA base"):
        encode_dna("ATGN")


def test_non_string_input():
    with pytest.raises(TypeError):
        encode_dna(123)


def test_sequence_with_multiple_bases():
    result = encode_dna("AATTGGCC")

    expected = np.array(
        [0, 0, 1, 1, 2, 2, 3, 3],
        dtype=np.uint8
    )

    np.testing.assert_array_equal(result, expected)
