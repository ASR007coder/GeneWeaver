
import numpy as np
from numba import cuda


# GPU kernel: Add two arrays
@cuda.jit
def add_arrays(a, b, result):
    index = cuda.grid(1)

    if index < result.size:
        result[index] = a[index] + b[index]


def main():

    # Check GPU availability
    if not cuda.is_available():
        print("CUDA GPU is not available.")
        return

    print("CUDA GPU detected successfully!")

    # Input arrays
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    b = np.array([1, 2, 3, 4], dtype=np.int32)

    # Allocate GPU memory
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.device_array_like(a)

    # Configure GPU execution
    threads_per_block = 4
    blocks_per_grid = 1

    # Execute calculation on GPU
    add_arrays[blocks_per_grid, threads_per_block](
        d_a, d_b, d_result
    )

    # Copy result back to CPU
    result = d_result.copy_to_host()

    print("Array A:", a)
    print("Array B:", b)
    print("GPU Result:", result)


if __name__ == "__main__":
    main()
