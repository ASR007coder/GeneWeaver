# GeneWeaver/backend/tests/test_cpu_alignment.py

from backend.cpu_alignment import compare_dna, count_mismatches


# Test 1: Exact DNA match
def test_exact_match():
    result = compare_dna("ATGC", "ATGC")
    assert len(result) == 1
    assert result[0]["mismatches"] == 0


# Test 2: One mismatch
def test_one_mismatch():
    result = compare_dna("ATGC", "AGGC", max_mismatches=1)
    assert len(result) == 1
    assert result[0]["mismatches"] == 1


# Test 3: Multiple mismatches
def test_multiple_mismatches():
    result = compare_dna("ATGC", "AGGT", max_mismatches=2)
    assert len(result) == 1
    assert result[0]["mismatches"] == 2


# Test 4: No exact match
def test_no_exact_match():
    result = compare_dna("ATGC", "CCCC")
    assert len(result) == 0


# Test 5: Target longer than genome
def test_target_longer_than_genome():
    result = compare_dna("ATGC", "AT")
    assert len(result) == 0


# Test 6: Genomic position tracking
def test_position_tracking():
    result = compare_dna(
        "ATGC",
        "AATGCTATGC",
        chromosome="chr1",
        start_position=10520
    )

    positions = [item["position"] for item in result]

    assert positions == [10521, 10526]


# Test 7: Mismatch counting directly
def test_count_mismatches():
    assert count_mismatches("ATGC", "AGGC") == 1
