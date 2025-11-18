# 2D Dynamic Programming Benchmark (dynprog.c)

## Overview

This benchmark implements a Smith-Waterman-like local sequence alignment algorithm, representing a classic **irregular parallel pattern** with **diagonal dependencies**. It's one of the most challenging parallelization problems in computational biology.

## Problem Characteristics

### Computational Pattern
- **Type**: 2D dynamic programming with cross-iteration dependencies
- **Dependencies**: Each cell `dp[i][j]` depends on:
  - `dp[i-1][j-1]` (diagonal)
  - `dp[i-1][j]` (top)
  - `dp[i][j-1]` (left)
- **Parallelism**: Anti-diagonal (wavefront) elements are independent
- **Challenge**: Exposing parallelism while respecting dependencies

### Memory Access Pattern
```
    j-1    j
i-1  ↘    ↓
  i   →   X

Each X depends on three previous values
```

### Scoring System
- **Match**: +2 points
- **Mismatch**: -1 point
- **Gap**: -1 point
- **Local alignment**: Negative scores reset to 0

## Parallelization Strategies

### Strategy 0: Serial Reference
**Description**: Classic sequential dynamic programming  
**Characteristics**:
- Simple nested loops
- Row-by-row computation
- Baseline for verification

**Code Pattern**:
```c
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
        dp[i][j] = max3(match, delete, insert);
    }
}
```

**Pros**: Simple, cache-friendly for small problems  
**Cons**: No parallelism, O(n²) time

---

### Strategy 1: Wavefront (Static Scheduling)
**Description**: Process anti-diagonals in parallel with static load distribution  
**Key Technique**: Anti-diagonal wavefront parallelization

**Code Pattern**:
```c
for (int wave = 2; wave <= 2*n; wave++) {
    #pragma omp parallel for schedule(static)
    for (int i = start_i; i <= end_i; i++) {
        int j = wave - i;
        // Compute dp[i][j]
    }
}
```

**Pros**:
- Exposes diagonal parallelism
- Simple to understand
- Predictable load distribution

**Cons**:
- Load imbalance (early/late waves have fewer elements)
- High synchronization overhead (barrier per wave)
- Poor cache locality

**Best For**: Regular problems, uniform computation per cell

---

### Strategy 2: Wavefront (Dynamic Scheduling)
**Description**: Wavefront with dynamic work distribution  
**Improvement**: Adaptive chunk sizing based on wave length

**Code Pattern**:
```c
int chunk = (end_i - start_i + 1) / (threads * 4);
#pragma omp parallel for schedule(dynamic, chunk)
```

**Pros**:
- Better load balancing than static
- Handles irregular work distribution
- Adapts to system load

**Cons**:
- Higher scheduling overhead
- Less predictable performance
- Still requires barrier per wave

**Best For**: Irregular workloads, variable cell computation costs

---

### Strategy 3: Tiled Wavefront
**Description**: Process tiles in wavefront order for cache locality  
**Key Innovation**: Combines blocking with wavefront

**Code Pattern**:
```c
// Process tile wavefronts
for (int tile_wave = 0; tile_wave < 2*num_tiles-1; tile_wave++) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ti = start_ti; ti <= end_ti; ti++) {
        // Process tile [ti][tj]
        for (int i = i_start; i <= i_end; i++) {
            #pragma omp simd
            for (int j = j_start; j <= j_end; j++) {
                // Compute with SIMD
            }
        }
    }
}
```

**Pros**:
- Excellent cache locality within tiles
- SIMD-friendly inner loops
- Reduces synchronization frequency
- Better memory bandwidth utilization

**Cons**:
- Complex implementation
- Tile size tuning required
- Overhead for small problems

**Best For**: Large problems, cache-sensitive applications

---

### Strategy 4: Task-Based Recursive
**Description**: Divide-and-conquer with OpenMP tasks  
**Approach**: Recursive quadrant subdivision

**Code Pattern**:
```c
#pragma omp task shared(dp) if(size > threshold)
compute_quadrant(top_left);
#pragma omp taskwait

#pragma omp task
compute_quadrant(top_right);
#pragma omp task
compute_quadrant(bottom_left);
#pragma omp taskwait

#pragma omp task
compute_quadrant(bottom_right);
```

**Dependency Order**:
1. Top-left (no dependencies)
2. Top-right and bottom-left (parallel, depend on top-left)
3. Bottom-right (depends on all others)

**Pros**:
- Automatic load balancing via task stealing
- Deep parallelism hierarchy
- Flexible granularity control
- Good for heterogeneous systems

**Cons**:
- Task creation overhead
- Complex dependency management
- Requires OpenMP 3.0+

**Best For**: Deep parallel systems, irregular computation patterns

---

### Strategy 5: Hybrid Striped
**Description**: Combines row striping with local wavefront  
**Innovation**: Reduces synchronization by processing row groups

**Code Pattern**:
```c
for (int stripe = 0; stripe < num_stripes; stripe++) {
    // Local wavefront within stripe
    for (int local_wave = 2; local_wave <= stripe_width+n; local_wave++) {
        #pragma omp parallel for schedule(dynamic)
        for (int local_i = 1; local_i <= stripe_width; local_i++) {
            // Compute with reduced synchronization
        }
    }
}
```

**Pros**:
- Reduces global synchronization
- Balances parallelism and locality
- Tunable stripe width
- Good middle ground

**Cons**:
- Still has synchronization overhead
- Stripe width tuning needed
- Not optimal for very large problems

**Best For**: Medium-sized problems, moderate thread counts

---

### Strategy 6: Ordered Directive (Doacross)
**Description**: OpenMP 4.5+ explicit dependency tracking  
**Feature**: `ordered(2)` with sink/source dependencies

**Code Pattern**:
```c
#pragma omp parallel for ordered(2) schedule(dynamic)
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
        #pragma omp ordered depend(sink: i-1,j) depend(sink: i,j-1)
        // Compute dp[i][j]
        #pragma omp ordered depend(source)
    }
}
```

**Pros**:
- Most portable approach
- Explicit dependency specification
- Compiler handles synchronization
- Works with any loop structure

**Cons**:
- Often serializes execution
- High overhead from fine-grained synchronization
- Limited speedup in practice
- Compiler-dependent performance

**Best For**: Prototyping, correctness verification, simple porting

---

## Performance Characteristics

### Expected Speedup Trends

| Strategy | Small (n=1000) | Medium (n=4000) | Large (n=10000) |
|----------|----------------|-----------------|-----------------|
| Wavefront Static | 2-4x | 4-8x | 6-12x |
| Wavefront Dynamic | 2-5x | 5-10x | 8-15x |
| Tiled | 3-6x | 8-16x | 12-25x |
| Task-based | 2-4x | 6-12x | 10-20x |
| Hybrid | 3-5x | 7-14x | 10-22x |
| Ordered | 1-2x | 1-3x | 1-4x |

### Cache Behavior

**Serial**: Good temporal locality, sequential access  
**Wavefront**: Poor locality (diagonal access pattern)  
**Tiled**: Excellent locality (blocked access)  
**Task**: Variable (depends on task granularity)

### Synchronization Overhead

**Barriers per iteration**:
- Wavefront: 2n barriers (one per anti-diagonal)
- Tiled: O(n/tile_size) barriers
- Task: No explicit barriers (implicit in taskwait)
- Ordered: Fine-grained (per cell)

## Compilation and Execution

### Basic Compilation
```bash
gcc -O3 -march=native -fopenmp -o dynprog dynprog.c -lm
```

### Intel Compiler
```bash
icc -O3 -xHost -qopenmp -o dynprog dynprog.c -lm
```

### Clang with OpenMP
```bash
clang -O3 -march=native -fopenmp -o dynprog dynprog.c -lm
```

### Execution Options
```bash
# Default size (4000), default threads
./dynprog

# Custom size, default threads
./dynprog 8000

# Custom size and thread count
./dynprog 8000 16

# Set threads via environment
export OMP_NUM_THREADS=8
./dynprog 4000
```

### Environment Variables
```bash
# Set thread count
export OMP_NUM_THREADS=16

# Thread affinity (spread threads)
export OMP_PROC_BIND=spread

# Place threads on cores
export OMP_PLACES=cores

# Dynamic adjustment
export OMP_DYNAMIC=false
```

## Tuning Guidelines

### Tile Size Selection
- **Small problems (n < 2000)**: tile_size = 32-64
- **Medium problems (n = 2000-8000)**: tile_size = 64-128
- **Large problems (n > 8000)**: tile_size = 128-256

**Rule of thumb**: tile_size × tile_size should fit in L1 cache

### Stripe Width
- **Default**: n / (4 × num_threads)
- **Memory-bound**: Smaller stripes (better locality)
- **Compute-bound**: Larger stripes (less overhead)

### Task Granularity
- **min_size = 50-100**: Good for most systems
- **Smaller**: More parallelism, higher overhead
- **Larger**: Less overhead, less parallelism

### Thread Affinity
```bash
# Best for wavefront (spread for memory bandwidth)
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Best for tiled (close for cache sharing)
export OMP_PROC_BIND=close
export OMP_PLACES=threads
```

## Advanced Optimizations

### SIMD Vectorization
Inner loops in tiled strategy use `#pragma omp simd` for vectorization:
```c
#pragma omp simd
for (int j = j_start; j <= j_end; j++) {
    // Compute with SIMD instructions
}
```

### Memory Alignment
Uses 64-byte alignment for cache line optimization:
```c
char *seq1 = aligned_alloc(64, n * sizeof(char));
int *dp = aligned_alloc(64, (n+1) * (n+1) * sizeof(int));
```

### False Sharing Prevention
- Tiles are sized to avoid cache line contention
- Dynamic scheduling helps distribute work
- Aligned memory reduces false sharing

## Verification

All strategies are verified against the serial reference:
```
Error: max|dp_test - dp_ref| < 1e-10
```

Final alignment score should match reference exactly.

## Output Interpretation

```
Strategy                          Time  Throughput    Final Score    Error
─────────────────────────────────────────────────────────────────────────
Wavefront (static)             2.1234 sec   7.53 Mcells/s  Score:  15432  Error: 0e+00
```

**Metrics**:
- **Time**: Wall-clock time in seconds
- **Throughput**: Million cells computed per second
- **Final Score**: Alignment score (should match reference)
- **Error**: Maximum absolute difference from reference (should be 0)

## Key Insights

1. **Wavefront is fundamental**: Anti-diagonal parallelization is the core technique

2. **Load balancing matters**: Early and late waves have fewer elements; dynamic scheduling helps

3. **Cache locality crucial**: Tiling dramatically improves performance on large problems

4. **Synchronization is expensive**: Each barrier costs ~1-10 microseconds; minimize barriers

5. **Task-based is flexible**: Good for irregular patterns but has overhead

6. **SIMD helps**: Inner loops benefit from vectorization in tiled approaches

7. **Ordered is portable but slow**: Good for correctness, not performance

## Common Issues and Solutions

### Issue: Poor scalability beyond 8 threads
**Cause**: Synchronization overhead dominates  
**Solution**: Use tiled or task-based approach

### Issue: Slower than serial for small problems
**Cause**: Parallel overhead exceeds benefit  
**Solution**: Use serial for n < 1000

### Issue: Incorrect results
**Cause**: Race condition in wavefront computation  
**Solution**: Verify dependencies are correct (wave computation)

### Issue: High memory usage
**Cause**: Multiple DP matrices for verification  
**Solution**: Remove verification for production runs

## References

1. **Smith-Waterman Algorithm**: Original sequence alignment paper
2. **OpenMP Common Core**: Wavefront and task-based patterns
3. **PolyBench**: Standard benchmark suite methodology
4. **Parallel Dynamic Programming**: Academic literature on dependency patterns

## Future Enhancements

- [ ] GPU offloading with OpenMP 4.5+ target directives
- [ ] Distributed memory version with MPI
- [ ] Space-optimized version (O(n) instead of O(n²))
- [ ] Profile-guided task granularity selection
- [ ] Auto-tuning for tile and stripe sizes
- [ ] NUMA-aware memory placement

---

**Author**: Generated based on OpenMP Common Core practices  
**Version**: 1.0  
**Date**: November 2025  
**License**: Open source for educational and research purposes
