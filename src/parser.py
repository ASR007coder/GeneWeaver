from Bio import SeqIO
import os

def parse_fasta(file_path):
    """
    Reads a FASTA file and extracts the chromosome name and DNA sequence.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    sequences = {}
    
    # SeqIO parses the FASTA file
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
        
    return sequences

if __name__ == "__main__":
    print("Parsing FASTA file...")
    
    # This automatically finds the folder where parser.py is located (src)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "chr21.fasta")
    
    parsed = parse_fasta(file_path)
    
    print("\n--- Results ---")
    for name, seq in parsed.items():
        print(f"Chromosome ID: {name}")
        print(f"Sequence Length: {len(seq)} base pairs")
        print(f"First 50 bases: {seq[:50]}...")