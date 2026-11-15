from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path: str | Path) -> list[tuple[str, str]]:
    """
    Read a FASTA file and return sequence IDs
    with their DNA sequences.
    """
    records = []

    for record in SeqIO.parse(str(file_path), "fasta"):
        records.append((record.id, str(record.seq).upper()))

    return records