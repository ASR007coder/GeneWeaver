import os
from cleaner import parse_and_clean
from chunker import chunk_sequence

def find_pam_sites(sequence):
    """
    Scans a DNA sequence and returns the exact positions of all 'NGG' PAM sites.
    N = A, C, G, or T. So we look for AGG, CGG, GGG, and TGG.
    """
    pam_sites = []
    
    # Iterate through the sequence, stopping 3 bases before the end
    for i in range(len(sequence) - 2):
        # Extract a 3-letter window
        window = sequence[i:i+3]
        
        # Check if it ends with 'GG' (which covers NGG)
        if window.endswith("GG"):
            pam_sites.append({
                "position": i,
                "sequence": window
            })
            
    return pam_sites

if __name__ == "__main__":
    print("Scanning for CRISPR PAM sites...")
    
    # 1. Load and clean the data (from Week 1)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "chr21.fasta")
    cleaned_data = parse_and_clean(file_path)
    chrom_name, full_sequence = list(cleaned_data.items())[0]
    
    # 2. Chunk it (from Week 1)
    chunks = chunk_sequence(full_sequence, chunk_size=10000)
    
    print(f"Scanning {len(chunks)} chunks...")
    
    # 3. Scan the first chunk for PAM sites
    first_chunk = chunks[0]
    chunk_seq = first_chunk["sequence"]
    chunk_start_pos = first_chunk["start_position"]
    
    pam_sites = find_pam_sites(chunk_seq)
    
    print(f"\n--- PAM Scan Results for Chunk 1 (Positions {chunk_start_pos} to {chunk_start_pos + len(chunk_seq)}) ---")
    print(f"Total PAM sites found in this chunk: {len(pam_sites)}")
    
    print("\nFirst 5 PAM sites found:")
    for site in pam_sites[:5]:
        # Add the chunk's start position to get the absolute genomic position
        absolute_pos = chunk_start_pos + site["position"]
        print(f"  Sequence: {site['sequence']} | Absolute Genomic Position: {absolute_pos}")
        