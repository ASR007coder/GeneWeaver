def find_mismatches(target_seq, genomic_seq):
    """
    Compares the target guide sequence with the genomic sequence.
    Returns the count of mismatches and their exact positions.
    """
    if len(target_seq) != len(genomic_seq):
        raise ValueError("Sequences must be the same length to compare.")
        
    mismatches = 0
    mismatch_positions = []
    
    # Iterate through both sequences at the same time
    for i in range(len(target_seq)):
        if target_seq[i] != genomic_seq[i]:
            mismatches += 1
            mismatch_positions.append({
                "position": i,
                "target_base": target_seq[i],
                "genomic_base": genomic_seq[i]
            })
            
    return {
        "target": target_seq,
        "genomic_sequence": genomic_seq,
        "mismatch_count": mismatches,
        "mismatch_positions": mismatch_positions
    }

if __name__ == "__main__":
    print("--- Mismatch Analysis Test ---")
    
    # Example from your PDF document
    target = "ATGC"
    genomic = "AGGC"
    
    print(f"Target:  {target}")
    print(f"Genomic: {genomic}")
    
    result = find_mismatches(target, genomic)
    
    print(f"\nTotal mismatches found: {result['mismatch_count']}")
    for m in result["mismatch_positions"]:
        print(f"  Position {m['position']}: Target '{m['target_base']}' != Genomic '{m['genomic_base']}'")