import os
from Bio import SeqIO

def clean_sequence(sequence):
    """
    Removes invalid characters (anything not A, C, T, G) 
    and converts everything to uppercase.
    """
    valid_bases = set("ACGT")
    # Convert to uppercase and filter out anything that isn't A, C, T, G
    cleaned = "".join([base for base in sequence.upper() if base in valid_bases])
    return cleaned

def parse_and_clean(file_path):
    """
    Parses the FASTA file and returns a dictionary of cleaned sequences.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    cleaned_sequences = {}
    
    for record in SeqIO.parse(file_path, "fasta"):
        raw_seq = str(record.seq)
        cleaned_seq = clean_sequence(raw_seq)
        cleaned_sequences[record.id] = cleaned_seq
        
    return cleaned_sequences

if __name__ == "__main__":
    print("Cleaning FASTA file...")
    
    # Find the exact folder we are in
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "chr21.fasta")
    
    cleaned_data = parse_and_clean(file_path)
    
    print("\n--- Cleaning Results ---")
    for name, seq in cleaned_data.items():
        print(f"Chromosome ID: {name}")
        print(f"Cleaned Sequence Length: {len(seq)} base pairs")
        print(f"First 50 cleaned bases: {seq[:50]}...")