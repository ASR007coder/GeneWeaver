
import pytest

from backend.alignment.chunk_gpu_alignment import (
    process_genomic_chunk
)


def test_chunk_gpu_matching_position():
    chunk = {
        "chromosome": "chr1",
        "sequence": "ATGCAGGC",
        "start_position": 100
    }

    results = process_genomic_chunk(
        chunk,
        "ATGC",
        max_mismatches=0
    )

    assert len(results) == 1
    assert results[0]["chromosome"] == "chr1"
    assert results[0]["position"] == 100
    assert results[0]["sequence"] == "ATGC"
    assert results[0]["mismatches"] == 0


def test_chunk_gpu_no_match():
    chunk = {
        "chromosome": "chr2",
        "sequence": "TTTTTTTT",
        "start_position": 500
    }

    results = process_genomic_chunk(
        chunk,
        "AAAA",
        max_mismatches=0
    )

    assert results == []


def test_chunk_gpu_target_longer_than_chunk():
    chunk = {
        "chromosome": "chr1",
        "sequence": "ATG",
        "start_position": 100
    }

    results = process_genomic_chunk(
        chunk,
        "ATGC",
        max_mismatches=0
    )

    assert results == []


def test_invalid_chunk():
    chunk = {
        "sequence": "ATGC",
        "start_position": 100
    }

    with pytest.raises(ValueError):
        process_genomic_chunk(chunk, "ATGC")
