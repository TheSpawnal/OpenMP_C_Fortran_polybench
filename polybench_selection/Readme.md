# OpenMP/C Multithreading Benchmark Suite

## Overview
This benchmark suite evaluates multithreading capacity and efficiency using five diverse computational patterns from PolyBench/C, each implemented with multiple parallelization strategies in OpenMP.

## Benchmarks

### 1. 2MM (Matrix Multiplication Chain)
**Computation**: D = α·A·B·C + β·D  
**Characteristics**: Compute-intensive, regular access patterns, coarse-grained parallelism

**Strategies**:
- **Basic Parallel**: Standard parallel for loops
- **Collapsed Loops**: Multi-dimensional loop parallelization
- **Tiled/Blocked**: Cache-optimized with blocking
- **SIMD Vectorization**: Explicit SIMD directives
- **Task-based**: Dynamic task decomposition

### 2. Cholesky Decomposition
**Computation**: A = L·L^T decomposition  
**Characteristics**: Irregular parallelism, data dependencies, synchronization-heavy

**Strategies**:
- **Column-wise**: Parallel column updates
- **Blocked Tasks**: Task-based with blocking
- **Right-looking**: Update trailing matrix
- **Left-looking**: Use previous columns
- **Recursive**: Divide-and-conquer approach

### 3. Jacobi-2D Stencil
**Computation**: Iterative 5-point stencil  
**Characteristics**: Memory-bound, nearest-neighbor communication, structured parallelism

**Strategies**:
- **Red-Black Ordering**: Checkerboard pattern updates
- **Wavefront**: Diagonal parallelization
- **Tiled**: Block decomposition with ghost zones
- **SIMD**: Vectorized computation
- **Hierarchical**: Nested parallelism
- **Temporal Blocking**: Time-tiled iteration

### 4. Correlation Matrix
**Computation**: Pearson correlation coefficients  
**Characteristics**: Reduction operations, memory-intensive, streaming access

**Strategies**:
- **Row-wise**: Parallel row computation
- **Tiled**: 2D cache blocking
- **SIMD**: Vectorized statistics
- **Task-based**: Dynamic task scheduling
- **Reduction-based**: Custom reduction operations
- **Column-major**: Optimized memory layout

### 5. Dynamic Programming (2D)
**Computation**: Sequence alignment (Needleman-Wunsch style)  
**Characteristics**: Complex dependencies, wavefront parallelism, load balancing challenges

**Strategies**:
- **Diagonal Wavefront**: Anti-diagonal processing
- **Anti-diagonal Tasks**: Block-based task parallelism
- **Pipeline**: Row-wise pipeline parallelism
- **Tiled Dependencies**: OpenMP dependency clauses
- **Hybrid**: Coarse+fine-grained parallelism
- **SIMD Wavefront**: Vectorized diagonal processing

## Compilation

### Basic Build
```bash
make all          # Build all benchmarks with -O3
make debug        # Debug build with -g
make profile      # Build with profiling support
```

### Compiler Options
- GCC: `gcc -O3 -march=native -fopenmp`
- Intel: `icc -O3 -xHost -qopenmp`
- Clang: `clang -O3 -march=native -fopenmp=libomp`

## Running Benchmarks

### Quick Test
```bash
make test         # Run with small problem sizes
```

### Full Benchmark Suite
```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh          # Standard sizes
./run_benchmarks.sh --large  # Include large problem sizes
```

### Individual Benchmarks
```bash
# 2MM: dimensions ni, nj, nk, nl
./benchmark_2mm 800 900 1000 1100

# Cholesky: matrix dimension n
./benchmark_cholesky 1500

# Jacobi-2D: grid size n, max iterations
./benchmark_jacobi2d 1024 1000

# Correlation: m data points, n variables
./benchmark_correlation 2000 500

# Dynamic Programming: sequence lengths m, n
./benchmark_dynprog 2000 2000
```

## Environment Variables

### OpenMP Settings
```bash
export OMP_NUM_THREADS=8        # Number of threads
export OMP_PROC_BIND=close      # Thread affinity
export OMP_PLACES=cores         # Thread placement
export OMP_SCHEDULE="dynamic,1" # Loop scheduling
```

### NUMA Settings
```bash
export OMP_PROC_BIND=spread     # For NUMA systems
export OMP_PLACES=sockets       # Distribute across sockets
```

## Performance Analysis

### Key Metrics
- **Execution Time**: Wall-clock time for each strategy
- **Speedup**: Sequential time / Parallel time
- **Efficiency**: Speedup / Number of threads
- **Scalability**: Performance vs thread count

### Expected Performance Patterns

1. **2MM**: Near-linear speedup up to core count
2. **Cholesky**: Limited by dependencies, 3-6x typical speedup
3. **Jacobi-2D**: Memory-bandwidth limited, moderate speedup
4. **Correlation**: Good speedup for computation, limited by memory
5. **Dynamic Programming**: Challenging, 2-4x typical speedup

## Optimization Tips

### Cache Optimization
- Tile sizes: 32-128 for L1, 256-512 for L2
- Align data to cache lines (64 bytes)
- Use `__restrict` pointers where applicable

### Thread Affinity
- Close binding for cache sharing
- Spread binding for memory bandwidth
- Monitor with `OMP_DISPLAY_ENV=TRUE`

### SIMD Optimization
- Ensure aligned memory allocation
- Use `#pragma omp simd` directives
- Check vectorization reports: `-fopt-info-vec`

## Benchmark Insights

### Compute-bound (2MM)
- Benefits from all parallelization strategies
- Tiling crucial for large matrices
- SIMD provides additional speedup

### Synchronization-heavy (Cholesky)
- Task-based approaches reduce idle time
- Block algorithms improve cache usage
- Recursive methods good for large problems

### Memory-bound (Jacobi-2D, Correlation)
- Limited by memory bandwidth
- Tiling and temporal blocking help
- NUMA-aware allocation important

### Irregular (Dynamic Programming)
- Wavefront parallelization essential
- Load balancing critical
- Hybrid approaches most effective

## Troubleshooting

### Poor Scaling
- Check thread affinity settings
- Verify no resource contention
- Profile for load imbalance

### Incorrect Results
- Race conditions in reductions
- Missing barriers/synchronization
- False sharing in shared arrays

### Performance Variability
- Disable CPU frequency scaling
- Pin threads to cores
- Warm up caches before timing

## References

1. PolyBench/C benchmark suite
2. OpenMP 4.5/5.0 specifications
3. "Patterns for Parallel Programming" - Mattson et al.
4. Intel/GCC OpenMP optimization guides

## License
Benchmark implementations inspired by PolyBench/C and OpenMP patterns from the provided project files.
