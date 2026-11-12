target = "ATGC"
genome = "AAGCTTGC"

target_length = len(target)

for i in range(len(genome) - target_length + 1):

    window = genome[i:i + target_length]

    mismatches = 0

    for j in range(target_length):

        if target[j] != window[j]:
            mismatches += 1

    print(
        f"Position: {i} | "
        f"Window: {window} | "
        f"Mismatches: {mismatches}"
    )