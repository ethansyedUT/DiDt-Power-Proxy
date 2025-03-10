#!/bin/bash
for ((i=1; i<=3; i++)); do
    echo "Iteration $i"i
    ./fma_benchmark 819200000 65536 1000000
done


