#!/bin/bash

# Quick test script for all benchmarks

echo "=== Testing OpenMP PolyBench Suite ==="
echo ""

# Build with SMALL size for quick testing
echo "Building benchmarks with SMALL problem size..."
make clean > /dev/null 2>&1
make SIZE=SMALL > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"
echo ""

# Test each benchmark with 4 threads
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

benchmarks="benchmark_2mm benchmark_3mm benchmark_cholesky benchmark_correlation benchmark_nussinov"

for bench in $benchmarks; do
    echo "Testing $bench..."
    if [ -x "./$bench" ]; then
        # Run benchmark and capture key output
        output=$(./$bench 2>&1 | grep -E "Sequential:|GFLOPS|Error" | head -3)
        
        # Check if it ran without crashing
        if [ $? -eq 0 ]; then
            echo "  ✓ $bench executed successfully"
            # Show sequential time
            echo "  $(echo "$output" | grep Sequential)"
        else
            echo "  ✗ $bench failed to execute"
        fi
    else
        echo "  ✗ $bench not found or not executable"
    fi
    echo ""
done

echo "=== Test Complete ==="
echo ""
echo "To run comprehensive benchmarks, use:"
echo "  ./run_benchmarks.sh"
echo ""
echo "To test specific sizes:"
echo "  make mini    # Tiny problems (< 1 second)"
echo "  make small   # Small problems (~seconds)"
echo "  make medium  # Medium problems (~minutes)"
echo "  make large   # Large problems (DAS-5 recommended)"