
import numpy as np
import pytest
from numba import cuda

from backend.alignment.dna_encoding import encode_dna
from backend.alignment.gpu_alignment import (
    GPUAlignmentWorkspace,
    compare_dna_gpu,
)


@pytest.fixture
def workspace():
    ws = GPUAlignmentWorkspace()
    yield ws
    ws.close()


def test_encoded_array_uses_uint8():
    encoded = encode_dna("ATGC" * 2500)

    assert encoded.dtype == np.uint8
    assert encoded.nbytes == 10000


def test_workspace_reuses_target_allocation(workspace):
    target = encode_dna("ATGC")

    first = workspace.get_target(target)
    second = workspace.get_target(target)

    # Both calls should return the same device allocation.
    assert first is second


def test_workspace_replaces_changed_target(workspace):
    first_target = encode_dna("ATGC")
    second_target = encode_dna("TTTT")

    first = workspace.get_target(first_target)
    second = workspace.get_target(second_target)

    assert second is not first


def test_workspace_cleanup(workspace):
    workspace.get_target(encode_dna("ATGC"))

    workspace.close()

    assert workspace._target_host is None
    assert workspace._target_device is None


@pytest.mark.parametrize("chunk_size", [1000, 5000, 10000])
def test_different_chunk_sizes(chunk_size, workspace):
    genome = "ATGC" * (chunk_size // 4)
    target = "ATGC"

    counts = compare_dna_gpu(
        genome,
        target,
        max_mismatches=0,
        workspace=workspace,
    )

    expected = np.zeros(
        len(genome) - len(target) + 1,
        dtype=np.int32,
    )

    for i in range(len(expected)):
        window = genome[i:i + len(target)]
        expected[i] = sum(
            a != b for a, b in zip(target, window)
        )

    np.testing.assert_array_equal(counts, expected)


def test_repeated_processing_with_same_workspace(workspace):
    target = "ATGC"

    first = compare_dna_gpu(
        "ATGCAGGC",
        target,
        workspace=workspace,
    )

    second = compare_dna_gpu(
        "TTTTATGC",
        target,
        workspace=workspace,
    )

    np.testing.assert_array_equal(
        first,
        np.array([0, 4, 4, 3, 1], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        second,
        np.array([3, 3, 3, 4, 0], dtype=np.int32),
    )


def test_negative_mismatch_threshold():
    with pytest.raises(ValueError):
        compare_dna_gpu(
            "ATGC",
            "ATGC",
            max_mismatches=-1,
        )


def test_non_integer_mismatch_threshold():
    with pytest.raises(TypeError):
        compare_dna_gpu(
            "ATGC",
            "ATGC",
            max_mismatches=1.5,
        )
