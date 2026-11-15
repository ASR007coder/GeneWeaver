def find_matches(
    reference_sequence: str,
    query_sequence: str,
    max_mismatches: int = 0
) -> list[tuple[int, int]]:
    """
    Find query sequences inside a reference sequence.

    Returns:
        List of (zero_based_position, mismatch_count).
    """
    results = []
    query_length = len(query_sequence)

    if query_length == 0 or query_length > len(reference_sequence):
        return results

    for position in range(len(reference_sequence) - query_length + 1):
        window = reference_sequence[position:position + query_length]

        mismatches = sum(
            a != b for a, b in zip(window, query_sequence)
        )

        if mismatches <= max_mismatches:
            results.append((position, mismatches))

    return results