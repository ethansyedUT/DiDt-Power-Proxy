#!/bin/bash
for ((i=1; i<=100; i++)); do
    echo "Iteration $i"
    ./tf32_gemm --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707
done

