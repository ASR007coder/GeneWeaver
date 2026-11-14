import os
from cleaner import parse_and_clean

def chunk_sequence(sequence, chunk_size=10000):
    """
    Splits a large DNA sequence into smaller chunks.
    Tracks the original genomic position (start and end).
    """
    chunks = []
    for i in range(0, len(sequence), chunk_size):
        chunk_seq = sequence[i:i + chunk_size]
        chunks.append({
            "start_position": i,
            "end_position": i + len(chunk_seq),
            "sequence": chunk_seq
        })
    return chunks

if __name__ == "__main__":
    print("Chunking DNA sequence...")
    
    # 1. Get the cleaned sequence from Day 3
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "chr21.fasta")
    
    # Note: We are importing the cleaning function we built yesterday!
    cleaned_data = parse_and_clean(file_path)
    
    # 2. Get the first chromosome in the file
    chrom_name, full_sequence = list(cleaned_data.items())[0]
    
    # 3. Chunk it into pieces of 10,000
    chunks = chunk_sequence(full_sequence, chunk_size=10000)
    
    print(f"\n--- Chunking Results for {chrom_name} ---")
    print(f"Total sequence length: {len(full_sequence)}")
    print(f"Total chunks created: {len(chunks)}")
    
    # 4. Show the first 3 chunks as an example
    print("\nPreview of first 3 chunks:")
    for i in range(3):
        c = chunks[i]
        print(f"Chunk {i+1}: Positions {c['start_position']} to {c['end_position']}")
        print(f"  Sequence: {c['sequence'][:20]}... (length: {len(c['sequence'])})")