def calculate_candidate_score(mismatch_count, mismatch_positions, pam_compatible=True):
    """
    Calculates a heuristic score for a CRISPR off-target candidate.
    Higher score = More dangerous.
    """
    # If it doesn't have a PAM site, the CRISPR enzyme won't bind.
    if not pam_compatible:
        return 0

    # Start with a base score of 100 (perfect match is highly dangerous)
    score = 100
    
    # Penalty for mismatch count
    score -= (mismatch_count * 15)
    
    # Penalty for mismatch position
    for pos in mismatch_positions:
        if pos < 10:
            score -= 10  # Strong penalty for early mismatch
        else:
            score -= 5   # Weaker penalty for late mismatch

    # Ensure the score stays between 0 and 100
    return max(0, min(100, score))

if __name__ == "__main__":
    print("--- Biological Scoring Test ---")
    
    # Test 1: Perfect match (0 mismatches)
    perfect_score = calculate_candidate_score(0, [])
    print(f"Perfect Match Score: {perfect_score} (Expected: 100)")
    
    # Test 2: 1 mismatch near the start (position 2)
    near_score = calculate_candidate_score(1, [2])
    print(f"1 Early Mismatch Score: {near_score} (Expected: 75)")
    
    # Test 3: 2 mismatches near the end (positions 15, 18)
    far_score = calculate_candidate_score(2, [15, 18])
    print(f"2 Late Mismatch Score: {far_score} (Expected: 60)")
    
    # Test 4: No PAM site
    no_pam_score = calculate_candidate_score(1, [2], pam_compatible=False)
    print(f"No PAM Score: {no_pam_score} (Expected: 0)")