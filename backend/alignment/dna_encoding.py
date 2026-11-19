# # GeneWeaver/backend/alignment/dna_encoding.py



import numpy as np


DNA_MAPPING = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3
}


def encode_dna(sequence):
    if not isinstance(sequence, str):
        raise TypeError("DNA sequence must be a string")

    sequence = sequence.upper()

    if not sequence:
        raise ValueError("DNA sequence cannot be empty")

    encoded = []

    for index, base in enumerate(sequence):
        if base not in DNA_MAPPING:
            raise ValueError(
                f"Invalid DNA base '{base}' at position {index}"
            )

        encoded.append(DNA_MAPPING[base])

    return np.array(encoded, dtype=np.uint8)


if __name__ == "__main__":
    target = "ATGC"

    encoded = encode_dna(target)

    print("Original DNA:", target)
    print("Encoded DNA:", encoded)
    print("Data type:", encoded.dtype)
