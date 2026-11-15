import os
import numpy as np
from cleaner import parse_and_clean
from chunker import chunk_sequence

# Dictionary to map DNA letters to numbers
# GPU engines usually map A=0, C=1, G=2, T=3
DNA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def encode_chunk(sequence):
    """
    Converts a string of DNA (e.g., 'ACGT') into a NumPy array of integers.
    """
    # We create an array of zeros with the same length as the sequence
    encoded = np.zeros(len(sequence), dtype=np.int8)
    
    for i, base in enumerate(sequence):
        encoded[i] = DNA_MAP.get(base, -1) # -1 for any unexpected character
        
    return encoded

if __name__ == "__main__":
    print("Encoding DNA sequence...")
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "chr21.fasta")
    
    # 1. Get the cleaned sequence (Day 3)
    cleaned_data = parse_and_clean(file_path)
    chrom_name, full_sequence = list(cleaned_data.items())[0]
    
    # 2. Chunk it (Day 4)
    chunks = chunk_sequence(full_sequence, chunk_size=10000)
    
    print(f"\n--- Encoding Results for {chrom_name} ---")
    print(f"Total chunks: {len(chunks)}")
    
    # 3. Encode the first chunk as a test
    first_chunk = chunks[0]
    chunk_seq = first_chunk["sequence"]
    start_pos = first_chunk["start_position"]
    
    encoded_array = encode_chunk(chunk_seq)
    
    print(f"\nChunk 1 (Positions {start_pos} to {start_pos + len(chunk_seq)}):")
    print(f"Original string (first 10): {chunk_seq[:10]}")
    print(f"Encoded integers (first 10): {encoded_array[:10]}")
    print(f"NumPy array shape: {encoded_array.shape}")