# GeneWeaver - Genomic Data Pipeline (Member 2)

This module handles the parsing, cleaning, chunking, and encoding of raw human DNA data (FASTA format) for GPU processing.

## Pipeline Stages
1. **Parser (`parser.py`):** Reads the raw `.fasta` file from NCBI.
2. **Cleaner (`cleaner.py`):** Removes invalid DNA bases (like 'N') and converts everything to uppercase.
3. **Chunker (`chunker.py`):** Splits the massive sequence into chunks of 10,000 bases and tracks their start/end positions.
4. **Encoder (`encoder.py`):** Converts DNA letters (A, C, T, G) into integer NumPy arrays (A=0, C=1, G=2, T=3) for the GPU.

## How to use this for the GPU (Member 1)
To get the encoded arrays for your CUDA kernels, run:
```python
from cleaner import parse_and_clean
from chunker import chunk_sequence
from encoder import encode_chunk

# 1. Load and clean
data = parse_and_clean("src/chr21.fasta")
seq = list(data.values())[0]

# 2. Chunk
chunks = chunk_sequence(seq, chunk_size=10000)

# 3. Encode
first_chunk_array = encode_chunk(chunks[0]["sequence"])