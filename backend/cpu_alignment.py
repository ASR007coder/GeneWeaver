# GeneWeaver/backend/cpu_alignment.py

def compare_dna(target, genome):
    target_length = len(target)
    results = []

    for i in range(len(genome) - target_length + 1):
        window = genome[i:i + target_length]

        if window == target:
            results.append({
                "position": i,
                "sequence": window
            })

    return results


# Test
target = "ATGC"
genome = "AATGCTATGC"

print(compare_dna(target, genome))
