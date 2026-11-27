# tests/test_pipeline.py
from src.parser import parse_fasta
from src.cleaner import clean_sequence
from src.chunker import chunk_sequence

def test_parser():
    # Add your actual parser test logic here if you have it
    pass

def test_cleaner():
    # Add your actual cleaner test logic here if you have it
    pass

def test_chunker_with_overlap():
    # 100 bases, chunk size 50, overlap 10
    seq = "A" * 100
    chunks = chunk_sequence(seq, chunk_size=50, overlap=10)
    
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    # Verify that the last 10 chars of chunk 1 match the first 10 chars of chunk 2
    assert chunks[0][-10:] == chunks[1][:10] 