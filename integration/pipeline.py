import csv
from pathlib import Path
from dataclasses import dataclass

from parser.fasta_parser import parse_fasta
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
    query_len = len(query_sequence)

    for position, mismatch_count in matches:
        matched_seq = reference_sequence[position : position + query_len]
        results.append(
            AlignmentResult(
                chromosome=chromosome,
                genomic_position=position,
                matched_sequence=matched_seq,
                mismatch_count=mismatch_count
            )
        )

    return results


def save_results_to_csv(
    results: list[AlignmentResult],
    output_file: str | Path
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as file:
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


def run_fasta_pipeline(
    fasta_file: str | Path,
    query_sequence: str,
    output_file: str | Path,
    max_mismatches: int = 0
) -> list[AlignmentResult]:
    records = parse_fasta(fasta_file)
    all_results = []

    for chromosome, reference_sequence in records:
        chr_results = run_alignment(
            chromosome=chromosome,
            reference_sequence=reference_sequence,
            query_sequence=query_sequence,
            max_mismatches=max_mismatches
        )
        all_results.extend(chr_results)

    save_results_to_csv(all_results, output_file)
    return all_results
