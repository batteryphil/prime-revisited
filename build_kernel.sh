#!/bin/bash

echo "Compiling PRIME C Inference Kernel..."

# -O3: Maximum optimization
# -march=native: Enable architecture-specific instructions (AVX2/AVX-512)
# -fopenmp: Enable OpenMP parallelization loops
# -ffast-math: Allow aggressive math optimizations (safe for our simple MAC loop)

gcc -O3 -march=native -mavx512f -mavx512bw -mavx512dq -fopenmp -ffast-math prime_kernel.c test_prime_kernel.c -o prime_benchmark -lm

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running benchmark..."
    echo ""
    ./prime_benchmark
else
    echo "Compilation failed."
fi
