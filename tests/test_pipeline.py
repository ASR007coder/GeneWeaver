import pytest
import sys
import os

# Add the 'src' folder to the Python path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cleaner import clean_sequence
from chunker import chunk_sequence
from encoder import encode_chunk

def test_cleaning():
    """Test that invalid characters (N, lowercase) are removed."""
    raw_dna = "ACGTNacgt"
    cleaned = clean_sequence(raw_dna)
    # Should only keep uppercase ACGT
    assert cleaned == "ACGTACGT"

def test_chunking():
    """Test that a sequence splits into correct sizes."""
    seq = "A" * 25000  # 25,000 bases
    chunks = chunk_sequence(seq, chunk_size=10000)
    
    # Should create 3 chunks (10000, 10000, 5000)
    assert len(chunks) == 3
    assert chunks[0]["start_position"] == 0
    assert chunks[1]["start_position"] == 10000
    assert len(chunks[2]["sequence"]) == 5000

def test_encoding():
    """Test that DNA letters map to integers correctly."""
    seq = "ACGT"
    encoded = encode_chunk(seq)
    
    # A=0, C=1, G=2, T=3
    assert list(encoded) == [0, 1, 2, 3]

def test_integration():
    """Test the full mini-pipeline: clean -> chunk -> encode."""
    raw_dna = "ACGTNNACGT" * 3000 # 30,000 bases with invalid Ns
    cleaned = clean_sequence(raw_dna) # 24,000 valid bases
    chunks = chunk_sequence(cleaned, chunk_size=10000) # 3 chunks
    
    assert len(chunks) == 3
    
    # Encode the first chunk
    encoded = encode_chunk(chunks[0]["sequence"])
    assert len(encoded) == 10000
    assert encoded[0] == 0 # First base should be A (0)