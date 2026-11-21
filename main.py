from integration.pipeline import run_fasta_pipeline


def main():
    print("================================")
    print("       GeneWeaver Engine")
    print("================================")

    fasta_path = "data/sample_genome.fasta"
    query_dna = "ATCG"
    output_csv = "results/alignment_results.csv"
    max_mismatches = 1

    print(f"Input FASTA: {fasta_path}")
    print(f"Query DNA:   {query_dna} (max mismatches: {max_mismatches})")

    results = run_fasta_pipeline(
        fasta_file=fasta_path,
        query_sequence=query_dna,
        output_file=output_csv,
        max_mismatches=max_mismatches
    )

    print(f"Total matches found: {len(results)}")
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()