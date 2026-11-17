from dataclasses import dataclass

from alignment.cpu_alignment import find_matches


@dataclass
class AlignmentResult:
    chromosome: str
    genomic_position: int
    matched_sequence: str
    mismatch_count: int


def run_alignment(
    chromosome: str,
    reference_sequence: str,
    query_sequence: str,
    max_mismatches: int = 0
) -> list[AlignmentResult]:

    matches = find_matches(
        reference_sequence,
        query_sequence,
        max_mismatches
    )

    results = []

    for position, mismatch_count in matches:
        results.append(
            AlignmentResult(
                chromosome=chromosome,
                genomic_position=position,
                matched_sequence=query_sequence,
                mismatch_count=mismatch_count
            )
        )

    return results