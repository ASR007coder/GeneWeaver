import csv
from pathlib import Path
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


def save_results_to_csv(
    results: list[AlignmentResult],
    output_file: str | Path
) -> None:
    """Save alignment results to a CSV file."""

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "chromosome",
            "genomic_position",
            "matched_sequence",
            "mismatch_count"
        ])

        for result in results:
            writer.writerow([
                result.chromosome,
                result.genomic_position,
                result.matched_sequence,
                result.mismatch_count
            ])
