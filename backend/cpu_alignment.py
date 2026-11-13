# GeneWeaver/backend/cpu_alignment.py

# Count mismatches between target DNA and a genomic window
def count_mismatches(target, window):
    mismatches = 0

    for i in range(len(target)):
        if target[i] != window[i]:
            mismatches += 1

    return mismatches


# Compare target DNA with genome using a sliding window
def compare_dna(target, genome, max_mismatches=0):
    target_length = len(target)
    results = []

    for i in range(len(genome) - target_length + 1):
        window = genome[i:i + target_length]

        mismatches = count_mismatches(target, window)

        if mismatches <= max_mismatches:
            results.append({
                "position": i,
                "sequence": window,
                "mismatches": mismatches
            })

    return results


# Test
target = "ATGC"
genome = "AATGCTATGC"

print("Exact matches:")
print(compare_dna(target, genome))

print("\nMatches allowing 1 mismatch:")
print(compare_dna(target, genome, max_mismatches=1))
