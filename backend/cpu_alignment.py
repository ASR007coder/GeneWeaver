# GeneWeaver/backend/cpu_alignment.py
# CPU DNA ALIGNMENT ALGORITHM

# INPUT:
# 1. Target DNA sequence
# 2. Genome sequence
# 3. Maximum allowed mismatches

# PROCESS:

# 1. Calculate the length of target DNA.

# 2. Start scanning the genome from position 0.

# 3. Extract a window equal to the target length.

# 4. Compare target DNA with the extracted window.

# 5. Count the number of mismatches.

# 6. If mismatches are within the allowed limit:
#       Store chromosome, position, matched sequence,
#       and mismatch count.

# 7. Move the window one position forward.

# 8. Repeat until all possible windows are checked.

# OUTPUT:
# List of matching genomic regions.