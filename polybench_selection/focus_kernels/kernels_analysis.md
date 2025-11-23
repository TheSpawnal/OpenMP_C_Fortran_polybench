# OpenMP PolyBench 2MM/3MM: Comprehensive Analysis & Optimization Guide

## Table of Contents
1. [Code Architecture Overview](#architecture)
2. [Strategy-by-Strategy Analysis](#strategies)
3. [Performance Optimization Insights](#optimizations)
4. [Compilation Commands](#compilation)
5. [Improvement Recommendations](#improvements)

---

## 1. Code Architecture Overview {#architecture}

### Memory Layout Strategy
Both implementations use **aligned memory allocation** (64-byte alignment):
```c
posix_memalign(&ptr, ALIGN_SIZE, size)
```
**Why this matters:**
- Enables efficient SIMD operations (AVX/AVX2 require 32/64-byte alignment)
- Reduces cache line splits
- Improves prefetcher effectiveness

### Problem Size Hierarchy
```
MINI    → Testing/debugging (very small)
SMALL   → Quick validation
MEDIUM  → Standard benchmarking
LARGE   → Stress testing
STANDARD→ Default (moderate size)
```

---

## 2. Strategy-by-Strategy Analysis {#strategies}

### **2MM Benchmark Strategies**

#### Strategy 1: Sequential Baseline
```c
void kernel_2mm_sequential(...)
```
**What it does:**
- E = α(AB)C + βD broken into two stages
- Stage 1: tmp = α * A * B
- Stage 2: D = tmp * C + β * D

**Key characteristics:**
- Three nested loops per multiplication (i, j, k)
- No parallelization overhead
- Serves as speedup reference
- Cache behavior depends on loop ordering (i-j-k means row-major access)

**Memory access pattern:**
- A: Row-wise (stride-1 in k loop) ✓ Good
- B: Column-wise (stride-nj in k loop) ✗ Poor
- tmp: Row-wise write ✓ Good

**Optimization notes:**
- The `tmp[i*nj + j] = 0.0` initialization could be hoisted outside
- Inner k-loop could benefit from unrolling

---

#### Strategy 2: Basic Parallel
```c
#pragma omp parallel for
for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
        double sum = 0.0;
        for (int k = 0; k < nk; k++) {
            sum += alpha * A[i*nk + k] * B[k*nj + j];
        }
        tmp[i*nj + j] = sum;
    }
}
```

**What it does:**
- Parallelizes outer i-loop
- Each thread processes complete rows independently
- Uses private accumulator `sum` to avoid false sharing

**Thread distribution:**
- Default scheduling: `schedule(static)` divides rows evenly
- No data dependencies between iterations
- Good for uniform workloads

**Performance characteristics:**
- **Pros:**
  - Simple, low overhead
  - Good cache locality (each thread gets consecutive rows)
  - No synchronization in computation
- **Cons:**
  - Parallelism limited by ni (number of rows)
  - Potential load imbalance if ni < num_threads
  - Doesn't address memory access pattern issues

---

#### Strategy 3: Collapsed Loops
```c
#pragma omp parallel for collapse(2)
for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
        // compute tmp[i][j]
    }
}
```

**What it does:**
- Creates single iteration space: ni × nj iterations
- Distributes work across (i,j) pairs instead of just i

**When it helps:**
- When ni is small (fewer than threads)
- Improves load balancing
- Increases parallelism granularity

**Technical details:**
- Compiler linearizes: `iteration = i * nj + j`
- Each thread gets chunk of linearized space
- Better utilization with small ni, large nj

**Performance trade-offs:**
- **Better:** When ni ≈ num_threads (improves balance)
- **Worse:** When ni >> num_threads (adds overhead)
- **Overhead:** Iteration space calculation, potential false sharing

---

#### Strategy 4: Tiled/Blocked
```c
const int TILE = 32;  // Optimized for cache
#pragma omp parallel for collapse(2) schedule(static)
for (int ii = 0; ii < ni; ii += TILE) {
    for (int jj = 0; jj < nj; jj += TILE) {
        for (int kk = 0; kk < nk; kk += TILE) {
            // Process TILE×TILE block
            for (int i = ii; i < i_end; i++) {
                for (int k = kk; k < k_end; k++) {
                    double aik = alpha * A[i*nk + k];
                    #pragma omp simd
                    for (int j = jj; j < j_end; j++) {
                        tmp[i*nj + j] += aik * B[k*nj + j];
                    }
                }
            }
        }
    }
}
```

**What it does:**
- Divides computation into cache-friendly TILE×TILE blocks
- Loop reordering: i-k-j instead of i-j-k
- Broadcasts A[i][k] across inner j loop

**Cache optimization strategy:**
- TILE=32 means 32×32×8 = 8KB per block (fits L1)
- Reuses data within tiles before eviction
- Reduces cache misses significantly

**Advanced technique:**
- `double aik = alpha * A[i*nk + k];` - hoisting multiplies
- `#pragma omp simd` - vectorizes innermost loop
- i-k-j ordering: Better for column-major B access

**Memory access improvements:**
- Original i-j-k: stride-nj access to B (bad)
- Tiled i-k-j: stride-1 access to B row (good!)
- Temporal locality: Reuses tmp[i][j] across k iterations

**Tuning parameters:**
- TILE size should fit in L1/L2 cache
- Your system: L1=32KB → TILE=32 is reasonable
- Could test TILE={16, 24, 32, 48, 64}

---

#### Strategy 5: SIMD Vectorization
```c
void kernel_2mm_simd(int ni, int nj, int nk, int nl,
                    double alpha, double beta,
                    double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ C, double *__restrict__ D,
                    double *__restrict__ tmp) {
    // Assume aligned pointers
    A = __builtin_assume_aligned(A, ALIGN_SIZE);
    B = __builtin_assume_aligned(B, ALIGN_SIZE);
    
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        #pragma omp simd aligned(tmp:ALIGN_SIZE)
        for (int j = 0; j < nj; j++) {
            tmp[i*nj + j] = 0.0;
        }
        
        for (int k = 0; k < nk; k++) {
            double aik = alpha * A[i*nk + k];
            #pragma omp simd aligned(B,tmp:ALIGN_SIZE)
            for (int j = 0; j < nj; j++) {
                tmp[i*nj + j] += aik * B[k*nj + j];
            }
        }
    }
}
```

**What it does:**
- Explicit SIMD directives for vectorization
- `__restrict__` prevents pointer aliasing
- `__builtin_assume_aligned` hints compiler about alignment

**SIMD mechanics:**
- AVX2: Processes 4 doubles simultaneously (256-bit registers)
- AVX-512: Processes 8 doubles simultaneously (512-bit registers)
- Compiler generates `vmovapd`, `vfmadd231pd` instructions

**Key optimizations:**
- `aligned(tmp:ALIGN_SIZE)` - enables aligned loads/stores
- `reduction(+:sum)` - parallel reduction for SIMD lanes
- Hoisted `aik` - broadcast scalar to vector register

**Performance expectations:**
- Theoretical speedup: 4× with AVX2, 8× with AVX-512
- Actual speedup: 2-3× (memory bandwidth limited)

---

#### Strategy 6: Task-based with Dependencies
```c
#pragma omp parallel
{
    #pragma omp single
    {
        for (int i = 0; i < ni; i += CHUNK) {
            for (int j = 0; j < nj; j += CHUNK) {
                #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
                {
                    // Compute tmp block
                }
            }
        }
        
        for (int i = 0; i < ni; i += CHUNK) {
            for (int j = 0; j < nl; j += CHUNK) {
                #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
                                 depend(inout:D[i*nl+j:CHUNK*CHUNK])
                {
                    // Compute D block
                }
            }
        }
    }
}
```

**What it does:**
- Creates fine-grained tasks (CHUNK×CHUNK blocks)
- Uses dependency clauses for automatic synchronization
- Runtime schedules tasks when dependencies satisfied

**Dependency graph:**
- Stage 1 tasks: Independent (can run in parallel)
- Stage 2 tasks: Depend on corresponding tmp rows
- Runtime maintains DAG (Directed Acyclic Graph)

**Advantages:**
- Dynamic load balancing
- Overlaps computation from both stages
- Handles irregular workloads well

**Overhead considerations:**
- Task creation cost: ~1-2μs per task
- Dependency tracking: O(num_tasks)
- Only beneficial if CHUNK > threshold (typically 32-64)

**Best for:**
- Irregular problem sizes
- Heterogeneous systems
- Complex dependency patterns

---

### **3MM Benchmark Strategies**

#### Strategy 7: Pipeline Parallel (3MM specific)
```c
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    // Stage 1: Compute E (my rows)
    int e_chunk = (ni + num_threads - 1) / num_threads;
    int e_start = tid * e_chunk;
    // ... compute E[e_start:e_end]
    
    // Stage 2: Compute F (my rows)
    int f_chunk = (nj + num_threads - 1) / num_threads;
    // ... compute F[f_start:f_end]
    
    #pragma omp barrier
    
    // Stage 3: Compute G (my rows)
    // ... compute G[g_start:g_end]
}
```

**What it does:**
- Each thread computes portions of E, F, G sequentially
- Exploits parallelism in both E=AB and F=CD (independent!)
- Barrier ensures E and F complete before G computation

**Key insight:**
- In 3MM, first two multiplications are independent
- Pipeline allows overlapping E and F computation
- Each thread stays on same cores (cache warm)

**Performance characteristics:**
- Less thread migration than `#pragma omp parallel for`
- Better cache locality (thread affinity)
- Trades off parallelism granularity for overhead reduction

**Load balancing:**
- Manual chunking: `(ni + num_threads - 1) / num_threads`
- Might have imbalance if ni % num_threads != 0
- Could use guided scheduling for better balance

---

## 3. Performance Optimization Insights {#optimizations}

### Memory Bandwidth Bottleneck
Your i5-10210U specs:
- 4 cores, 8 threads (Hyperthreading)
- L1: 32KB per core
- L2: 256KB per core  
- L3: 6MB shared
- Memory bandwidth: ~40 GB/s

**Implications:**
- Matrix operations are memory-bound for large sizes
- Bandwidth per core: ~10 GB/s
- SIMD can't help if already bandwidth-limited
- Cache blocking is CRITICAL

### NUMA Considerations (Not applicable to your system)
Single-socket system = uniform memory access
Multi-socket would require:
- `numa_set_preferred()` calls
- Thread pinning with `OMP_PLACES`

### False Sharing Prevention
Your code correctly uses **private accumulators**:
```c
double sum = 0.0;  // Private to each thread
for (int k = 0; k < nk; k++) {
    sum += ...;
}
tmp[i*nj + j] = sum;  // Single write
```

**Why this matters:**
- Cache line = 64 bytes = 8 doubles
- Without private sum: Each addition would cause cache coherence traffic
- With private sum: Only final write touches shared memory

---

## 4. Compilation Commands {#compilation}

### Basic Compilation
```bash
# Standard build
gcc -fopenmp -O3 -march=native benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm

# With specific size
gcc -fopenmp -O3 -march=native -DLARGE benchmark_2mm.c benchmark_metrics.c -o bench_2mm_large -lm
```

### Aggressive Optimization Levels
```bash
# Level 1: Safe optimizations
gcc -fopenmp -O3 -march=native -mtune=native \
    -ffast-math -funroll-loops \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm

# Level 2: Aggressive + Vectorization info
gcc -fopenmp -O3 -march=native -mtune=native \
    -ffast-math -funroll-loops -ftree-vectorize \
    -fopt-info-vec-optimized -fopt-info-vec-missed \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm

# Level 3: Maximum optimization (may break precision)
gcc -fopenmp -Ofast -march=native -mtune=native \
    -funroll-loops -fprefetch-loop-arrays \
    -fno-signed-zeros -fno-trapping-math \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm
```

### Architecture-Specific Flags
```bash
# For your i5-10210U (Comet Lake - AVX2 support)
gcc -fopenmp -O3 -march=comet-lake -mavx2 -mfma \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm

# Alternatively, let GCC detect
gcc -fopenmp -O3 -march=native -mtune=native \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm
```

### Debugging & Analysis Builds
```bash
# Debug with optimization
gcc -fopenmp -O2 -g -march=native \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm_debug -lm

# Generate assembly for inspection
gcc -fopenmp -O3 -march=native -S \
    -fverbose-asm benchmark_2mm.c -o bench_2mm.s

# Generate optimization report
gcc -fopenmp -O3 -march=native -fopt-info-all=opt_report.txt \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm
```

### Profile-Guided Optimization (PGO)
```bash
# Step 1: Compile with instrumentation
gcc -fopenmp -O3 -march=native -fprofile-generate \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm_prof -lm

# Step 2: Run to collect profile data
./bench_2mm_prof

# Step 3: Recompile with profile data
gcc -fopenmp -O3 -march=native -fprofile-use \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm_pgo -lm
```

### Link-Time Optimization (LTO)
```bash
# Single-file LTO
gcc -fopenmp -O3 -march=native -flto \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm

# Multi-stage LTO
gcc -fopenmp -O3 -march=native -flto -c benchmark_2mm.c -o benchmark_2mm.o
gcc -fopenmp -O3 -march=native -flto -c benchmark_metrics.c -o benchmark_metrics.o
gcc -fopenmp -O3 -march=native -flto benchmark_2mm.o benchmark_metrics.o -o bench_2mm -lm
```

### Sanitizer Builds (Correctness Checking)
```bash
# Thread sanitizer (detects data races)
gcc -fopenmp -O1 -g -fsanitize=thread \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm_tsan -lm

# Address sanitizer (detects memory errors)
gcc -fopenmp -O1 -g -fsanitize=address \
    benchmark_2mm.c benchmark_metrics.c -o bench_2mm_asan -lm
```

### Complete Makefile Example
```makefile
CC = gcc
CFLAGS = -fopenmp -O3 -march=native -mtune=native -Wall -Wextra
LDFLAGS = -lm

SIZES = MINI SMALL MEDIUM LARGE
BENCHMARKS = 2mm 3mm

all: $(foreach size,$(SIZES),$(foreach bench,$(BENCHMARKS),bench_$(bench)_$(size)))

bench_%_MINI: benchmark_%.c benchmark_metrics.c
	$(CC) $(CFLAGS) -DMINI $^ -o $@ $(LDFLAGS)

bench_%_SMALL: benchmark_%.c benchmark_metrics.c
	$(CC) $(CFLAGS) -DSMALL $^ -o $@ $(LDFLAGS)

bench_%_MEDIUM: benchmark_%.c benchmark_metrics.c
	$(CC) $(CFLAGS) -DMEDIUM $^ -o $@ $(LDFLAGS)

bench_%_LARGE: benchmark_%.c benchmark_metrics.c
	$(CC) $(CFLAGS) -DLARGE $^ -o $@ $(LDFLAGS)

clean:
	rm -f bench_* *.o *.gcda *.gcno

.PHONY: all clean
```

---

## 5. Improvement Recommendations {#improvements}

### Critical Issues Found

#### 1. **Missing Hierarchical Strategy (2MM)**
Your 2mm.c header mentions "Hierarchical (nested parallelism)" but it's not implemented.

**Implementation suggestion:**
```c
void kernel_2mm_hierarchical(int ni, int nj, int nk, int nl,
                             double alpha, double beta,
                             double *A, double *B, double *C, double *D,
                             double *tmp) {
    omp_set_nested(1);  // Enable nested parallelism
    omp_set_max_active_levels(2);
    
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        
        if (tid == 0) {
            // Team 0: Compute tmp = alpha * A * B
            #pragma omp parallel for collapse(2) num_threads(4)
            for (int i = 0; i < ni; i++) {
                for (int j = 0; j < nj; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < nk; k++) {
                        sum += alpha * A[i*nk + k] * B[k*nj + j];
                    }
                    tmp[i*nj + j] = sum;
                }
            }
        } else {
            // Team 1: Could prepare D scaling in parallel
            // (Limited opportunity here, but demonstrates concept)
        }
    }
    
    #pragma omp barrier
    
    // Second multiplication with full thread team
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i*nl + j];
            for (int k = 0; k < nj; k++) {
                sum += tmp[i*nj + k] * C[k*nl + j];
            }
            D[i*nl + j] = sum;
        }
    }
}
```

#### 2. **Cache-Optimized with Prefetching Strategy Missing**
Header mentions it but not implemented.

**Implementation:**
```c
void kernel_2mm_prefetch(int ni, int nj, int nk, int nl,
                        double alpha, double beta,
                        double *A, double *B, double *C, double *D,
                        double *tmp) {
    const int TILE = 32;
    const int PREFETCH_DISTANCE = 8;  // Tune this
    
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < ni; ii += TILE) {
        for (int jj = 0; jj < nj; jj += TILE) {
            for (int kk = 0; kk < nk; kk += TILE) {
                int i_end = (ii + TILE < ni) ? ii + TILE : ni;
                int j_end = (jj + TILE < nj) ? jj + TILE : nj;
                int k_end = (kk + TILE < nk) ? kk + TILE : nk;
                
                for (int i = ii; i < i_end; i++) {
                    // Prefetch next row of A
                    if (i + PREFETCH_DISTANCE < i_end) {
                        __builtin_prefetch(&A[(i+PREFETCH_DISTANCE)*nk + kk], 0, 3);
                    }
                    
                    for (int k = kk; k < k_end; k++) {
                        double aik = alpha * A[i*nk + k];
                        
                        // Prefetch next block of B
                        if (k + PREFETCH_DISTANCE < k_end) {
                            __builtin_prefetch(&B[(k+PREFETCH_DISTANCE)*nj + jj], 0, 3);
                        }
                        
                        #pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            tmp[i*nj + j] += aik * B[k*nj + j];
                        }
                    }
                }
            }
        }
    }
    
    // Similar for second multiplication...
}
```

#### 3. **3MM Missing "Hierarchical with Prefetching"**
Only 6 strategies implemented instead of 8.

#### 4. **FLOPS Calculation Issue in 2MM**
Your formula is slightly off:
```c
// Current (INCORRECT):
return 2LL * ni * nj * nk + 2LL * ni * nl * nj;

// Should be:
// Stage 1: tmp = alpha * A * B
//   - ni * nj * nk multiply-adds (2 ops each)
//   - ni * nj multiplies by alpha (1 op each)
// Stage 2: D = tmp * C + beta * D
//   - ni * nl * nj multiply-adds (2 ops each)
//   - ni * nl multiplies by beta (1 op each)
return ni * nj * (2LL * nk + 1) + ni * nl * (2LL * nj + 1);
```

### Code Quality Improvements

#### 1. **Add Runtime Tile Size Selection**
```c
int select_optimal_tile(int cache_size_kb, size_t element_size) {
    // L1 cache-aware tiling
    int max_tile = (int)sqrt(cache_size_kb * 1024 / element_size);
    // Round to power of 2
    int tile = 16;
    while (tile < max_tile && tile < 128) tile *= 2;
    return tile;
}
```

#### 2. **Add Thread Affinity Control**
```c
void set_thread_affinity() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
}
```

#### 3. **Enhanced Verification**
```c
static void verify_result_detailed(int ni, int nl, double *D_ref, double *D_test) {
    double max_error = 0.0;
    double avg_error = 0.0;
    int error_count = 0;
    
    for (int i = 0; i < ni * nl; i++) {
        double error = fabs(D_ref[i] - D_test[i]);
        avg_error += error;
        if (error > 1e-10) error_count++;
        if (error > max_error) max_error = error;
    }
    
    printf("Verification: max_err=%.2e, avg_err=%.2e, errors=%d/%d\n",
           max_error, avg_error / (ni*nl), error_count, ni*nl);
}
```

#### 4. **Add Memory Bandwidth Measurement**
```c
double measure_memory_bandwidth(int ni, int nj, int nk, double time) {
    // Bytes read: A (ni*nk) + B (nk*nj*ni)
    // Bytes written: tmp (ni*nj)
    size_t bytes_read = (size_t)ni * nk * sizeof(double) +
                        (size_t)ni * nk * nj * sizeof(double);
    size_t bytes_written = (size_t)ni * nj * sizeof(double);
    size_t total_bytes = bytes_read + bytes_written;
    
    return (total_bytes / (1024.0 * 1024.0 * 1024.0)) / time;  // GB/s
}
```

### Experimental Strategies to Add

#### 1. **Hybrid Task + SIMD**
```c
void kernel_2mm_hybrid(int ni, int nj, int nk, int nl,
                      double alpha, double beta,
                      double *A, double *B, double *C, double *D,
                      double *tmp) {
    const int TASK_CHUNK = 64;
    
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < ni; i += TASK_CHUNK) {
            #pragma omp task
            {
                int i_end = (i + TASK_CHUNK < ni) ? i + TASK_CHUNK : ni;
                for (int ii = i; ii < i_end; ii++) {
                    #pragma omp simd
                    for (int j = 0; j < nj; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < nk; k++) {
                            sum += alpha * A[ii*nk + k] * B[k*nj + j];
                        }
                        tmp[ii*nj + j] = sum;
                    }
                }
            }
        }
        #pragma omp taskwait
    }
    
    // Similar for second stage...
}
```

#### 2. **Adaptive Strategy Selection**
```c
typedef enum {
    STRATEGY_BASIC,
    STRATEGY_COLLAPSED,
    STRATEGY_TILED,
    STRATEGY_SIMD,
    STRATEGY_TASK
} StrategyType;

StrategyType select_strategy(int ni, int nj, int nk, int num_threads) {
    // Heuristics based on problem size
    if (ni * nj * nk < 1000000) return STRATEGY_BASIC;
    if (ni < num_threads) return STRATEGY_COLLAPSED;
    if (nk > 1000) return STRATEGY_TILED;
    return STRATEGY_SIMD;
}
```

### Testing Recommendations

#### Performance Testing Script
```bash
#!/bin/bash

echo "Running systematic performance tests..."

for size in MINI SMALL MEDIUM LARGE; do
    echo "=== Testing $size ==="
    gcc -fopenmp -O3 -march=native -D$size \
        benchmark_2mm.c benchmark_metrics.c -o bench_2mm -lm
    
    for threads in 1 2 4 8; do
        export OMP_NUM_THREADS=$threads
        echo "Threads: $threads"
        ./bench_2mm | grep -E "(Sequential|Speedup|GFLOPS)"
    done
done
```

#### Validation Script
```bash
#!/bin/bash

# Compile with different optimization levels
for opt in O0 O1 O2 O3 Ofast; do
    gcc -fopenmp -$opt -march=native \
        benchmark_2mm.c benchmark_metrics.c -o bench_2mm_$opt -lm
    echo "Testing -$opt..."
    ./bench_2mm_$opt > results_$opt.txt
done

# Compare results
python3 -c "
import sys
import re

def parse_results(filename):
    with open(filename) as f:
        content = f.read()
        times = re.findall(r'Time.*?(\d+\.\d+)', content)
        return [float(t) for t in times]

for opt in ['O0', 'O1', 'O2', 'O3', 'Ofast']:
    times = parse_results(f'results_{opt}.txt')
    print(f'{opt}: {min(times):.4f}s (best)')
"
```

---

## Summary of Key Findings

### Strengths ✓
1. **Well-structured code** with clear strategy separation
2. **Comprehensive strategies** covering major parallelization techniques
3. **Proper verification** and error checking
4. **Aligned memory allocation** for SIMD
5. **Good use of OpenMP features** (collapse, SIMD, tasks, dependencies)

### Areas for Improvement ⚠
1. **Missing implementations**: Hierarchical and prefetch strategies
2. **FLOPS calculation**: Slightly incorrect formula
3. **No dynamic tuning**: Fixed TILE sizes, no adaptation
4. **Limited metrics**: Could add bandwidth, cache misses (with PAPI)
5. **No architecture detection**: Could adjust strategies based on CPU features

### Performance Expectations on Your i5-10210U
- **Sequential**: ~5-10 GFLOPS (memory-bound)
- **Basic Parallel (8 threads)**: ~2.5-4× speedup
- **Tiled**: ~10-20% better than basic (cache effects)
- **SIMD**: ~1.5-2× over tiled (AVX2 utilization)
- **Hybrid Best**: ~5-6× speedup over sequential

### Next Steps
1. Implement missing strategies
2. Run systematic benchmarks across all sizes
3. Profile with `perf` to identify bottlenecks
4. Test with PAPI for hardware counters
5. Consider GPU offloading for very large sizes (OpenMP target)