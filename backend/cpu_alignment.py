# GeneWeaver/backend/cpu_alignment.py

# Count mismatches between target DNA and a genomic window
def count_mismatches(target, window):
    mismatches = 0

    for i in range(len(target)):
        if target[i] != window[i]:
            mismatches += 1

    return mismatches


# Compare target DNA with genome using a sliding window
def compare_dna(
    target,
    genome,
    chromosome="chr1",
    start_position=0,
    max_mismatches=0
):
    target_length = len(target)
    results = []

    for i in range(len(genome) - target_length + 1):
        window = genome[i:i + target_length]

        mismatches = count_mismatches(target, window)

        if mismatches <= max_mismatches:
            results.append({
                "chromosome": chromosome,
                "position": start_position + i,
                "sequence": window,
                "mismatches": mismatches
            })

    return results


# Test 1: Complete genome sequence
target = "ATGC"
genome = "AATGCTATGC"

results = compare_dna(
    target,
    genome,
    chromosome="chr1",
    start_position=0,
    max_mismatches=0
)

print("Complete genome results:")
for result in results:
    print(result)


# Test 2: Genome chunk with original starting position
chunk = "AATGCTATGC"

chunk_results = compare_dna(
    target,
    chunk,
    chromosome="chr1",
    start_position=10520,
    max_mismatches=0
)

print("\nChunk results:")
for result in chunk_results:
    print(result)
