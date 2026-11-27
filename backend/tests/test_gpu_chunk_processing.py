
import pytest
from numba import cuda

from backend.alignment.gpu_alignment import (
    GPUAlignmentWorkspace,
    process_chunk_gpu
)


@pytest.fixture
def workspace():
    """Create and clean up a GPU workspace."""

    if not cuda.is_available():
        pytest.skip("CUDA GPU is not available")

    gpu_workspace = GPUAlignmentWorkspace()

    yield gpu_workspace

    gpu_workspace.close()


# --------------------------------------------------
# 1. Test valid genomic chunk processing
# --------------------------------------------------

def test_process_chunk_gpu_valid_chunk(workspace):

    chunk = {
        "chromosome": "chr1",
        "sequence": "ATGCAGGC",
        "start_position": 1000
    }

    results = process_chunk_gpu(
        chunk=chunk,
        target_sequence="ATGC",
        max_mismatches=0,
        workspace=workspace
    )

    # Only one exact match exists.
    assert len(results) == 1

    assert results[0]["chromosome"] == "chr1"
    assert results[0]["position"] == 1000
    assert results[0]["sequence"] == "ATGC"
    assert results[0]["mismatches"] == 0


# --------------------------------------------------
# 2. Test chromosome metadata preservation
# --------------------------------------------------

def test_process_chunk_gpu_preserves_chromosome(workspace):

    chunk = {
        "chromosome": "chrX",
        "sequence": "ATGCAGGC",
        "start_position": 5000
    }

    results = process_chunk_gpu(
        chunk,
        target_sequence="ATGC",
        workspace=workspace
    )

    assert len(results) == 1

    assert all(
        result["chromosome"] == "chrX"
        for result in results
    )


# --------------------------------------------------
# 3. Test original genomic position tracking
# --------------------------------------------------

def test_process_chunk_gpu_preserves_positions(workspace):

    chunk = {
        "chromosome": "chr2",
        "sequence": "ATGCAGGC",
        "start_position": 10500
    }

    results = process_chunk_gpu(
        chunk,
        target_sequence="ATGC",
        workspace=workspace
    )

    positions = [
        result["position"]
        for result in results
    ]

    assert positions == [10500]


# --------------------------------------------------
# 4. Test mismatch threshold
# --------------------------------------------------

def test_process_chunk_gpu_mismatch_threshold(workspace):

    chunk = {
        "chromosome": "chr1",
        "sequence": "ATGTAGGC",
        "start_position": 100
    }

    results = process_chunk_gpu(
        chunk,
        target_sequence="ATGC",
        max_mismatches=1,
        workspace=workspace
    )

    # ATGT and AGGC both have one mismatch.
    assert len(results) == 2

    assert results[0]["position"] == 100
    assert results[0]["sequence"] == "ATGT"
    assert results[0]["mismatches"] == 1

    assert results[1]["position"] == 104
    assert results[1]["sequence"] == "AGGC"
    assert results[1]["mismatches"] == 1


# --------------------------------------------------
# 5. Test target longer than genomic chunk
# --------------------------------------------------

def test_process_chunk_gpu_target_longer_than_chunk(workspace):

    chunk = {
        "chromosome": "chr1",
        "sequence": "AT",
        "start_position": 100
    }

    results = process_chunk_gpu(
        chunk,
        target_sequence="ATGC",
        workspace=workspace
    )

    assert results == []


# --------------------------------------------------
# 6. Test invalid chunk type
# --------------------------------------------------

def test_process_chunk_gpu_invalid_chunk_type():

    with pytest.raises(TypeError):
        process_chunk_gpu(
            chunk="ATGCAGGC",
            target_sequence="ATGC"
        )


# --------------------------------------------------
# 7. Test missing chunk fields
# --------------------------------------------------

def test_process_chunk_gpu_missing_fields():

    chunk = {
        "chromosome": "chr1",
        "sequence": "ATGCAGGC"
    }

    with pytest.raises(ValueError):
        process_chunk_gpu(
            chunk,
            target_sequence="ATGC"
        )


# --------------------------------------------------
# 8. Test invalid chromosome
# --------------------------------------------------

def test_process_chunk_gpu_invalid_chromosome():

    chunk = {
        "chromosome": "",
        "sequence": "ATGCAGGC",
        "start_position": 100
    }

    with pytest.raises(ValueError):
        process_chunk_gpu(
            chunk,
            target_sequence="ATGC"
        )


# --------------------------------------------------
# 9. Test invalid genomic position
# --------------------------------------------------

def test_process_chunk_gpu_invalid_position():

    chunk = {
        "chromosome": "chr1",
        "sequence": "ATGCAGGC",
        "start_position": -100
    }

    with pytest.raises(ValueError):
        process_chunk_gpu(
            chunk,
            target_sequence="ATGC"
        )


# --------------------------------------------------
# 10. Test workspace reuse across chunks
# --------------------------------------------------

def test_gpu_workspace_reuse_across_chunks(workspace):

    chunk1 = {
        "chromosome": "chr1",
        "sequence": "ATGCAGGC",
        "start_position": 1000
    }

    chunk2 = {
        "chromosome": "chr2",
        "sequence": "TTATGC",
        "start_position": 5000
    }

    results1 = process_chunk_gpu(
        chunk1,
        target_sequence="ATGC",
        workspace=workspace
    )

    results2 = process_chunk_gpu(
        chunk2,
        target_sequence="ATGC",
        workspace=workspace
    )

    # Verify both chunks were processed.
    assert len(results1) == 1
    assert len(results2) == 1

    # Verify chromosome metadata.
    assert results1[0]["chromosome"] == "chr1"
    assert results2[0]["chromosome"] == "chr2"

    # Verify original genomic positions.
    assert results1[0]["position"] == 1000
    assert results2[0]["position"] == 5002

    # Verify mismatch counts.
    assert results1[0]["mismatches"] == 0
    assert results2[0]["mismatches"] == 0
