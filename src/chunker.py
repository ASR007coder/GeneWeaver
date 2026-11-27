# src/chunker.py

def chunk_sequence(sequence, chunk_size=10000, overlap=20):
    """
    Splits a DNA sequence into chunks of a specific size.
    Includes an overlap between adjacent chunks so matches aren't missed at the edges.
    """
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap size.")
        
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(sequence), step):
        # Grab the chunk
        chunk = sequence[i:i + chunk_size]
        
        # Stop if we reach the end of the sequence
        if not chunk:
            break
            
        chunks.append(chunk)
        
        # If the chunk is smaller than the full size, we've reached the end
        if len(chunk) < chunk_size:
            break
            
    return chunks

if __name__ == "__main__":
    # Quick test
    test_seq = "A" * 10050
    result = chunk_sequence(test_seq, chunk_size=10000, overlap=20)
    print(f"Generated {len(result)} chunks.")