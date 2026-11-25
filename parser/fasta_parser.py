
from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path: str | Path) -> list[tuple[str, str]]:
    """
    Read a FASTA file and return sequence IDs
    with their DNA sequences.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    records = []

    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            records.append((record.id, str(record.seq).upper()))

    return records
