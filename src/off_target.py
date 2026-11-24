# src/off_target.py

def find_off_targets(guide_rna, genome_chunks, max_mismatches=3):
    """
    Scans the genome chunks to find potential off-target binding sites.
    An off-target is a sequence that matches the guide RNA with <= max_mismatches.
    """
    off_targets = []
    
    # We simulate a basic search algorithm
    for chunk_index, chunk in enumerate(genome_chunks):
        # In a real scenario, this would use a sliding window (e.g., k-mer matching)
        # For this pipeline, we do a simplified check
        if _check_sequence(chunk, guide_rna, max_mismatches):
            off_targets.append({
                "chunk_id": chunk_index,
                "sequence": chunk,
                "mismatches": _count_mismatches(chunk, guide_rna)
            })
            
    return off_targets

def _check_sequence(sequence, guide_rna, max_mismatches):
    # Dummy logic for pipeline testing: just checking the first N bases
    # In reality, this would use a sliding window
    window = sequence[:len(guide_rna)]
    mismatches = _count_mismatches(window, guide_rna)
    return mismatches <= max_mismatches

def _count_mismatches(seq1, seq2):
    return sum(1 for a, b in zip(seq1, seq2) if a != b)

if __name__ == "__main__":
    print("Off-target module ready.")