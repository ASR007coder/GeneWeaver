"""
GeneWeaver - Main Application
"""

from parser.fasta_parser import parse_fasta
from alignment.cpu_alignment import find_matches


def main():
    print("================================")
    print("       GeneWeaver Engine")
    print("================================")

    print("FASTA Parser: Connected")
    print("CPU Alignment Engine: Connected")
    print("Application started successfully.")


if __name__ == "__main__":
    main()