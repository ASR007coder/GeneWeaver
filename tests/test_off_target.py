# tests/test_off_target.py
from src.off_target import find_off_targets

def test_find_off_targets():
    dummy_guide = "ACGTACGT"
    dummy_chunks = [
        "ACGTACGT", # Exact match (0 mismatches)
        "ACGTACGA", # 1 mismatch
        "TTTTTTTT", # 8 mismatches
    ]
    
    # Should only find the first two because max_mismatches is 3
    results = find_off_targets(dummy_guide, dummy_chunks, max_mismatches=3)
    
    assert len(results) == 2
    assert results[0]["mismatches"] == 0
    assert results[1]["mismatches"] == 1