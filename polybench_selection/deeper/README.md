# OpenMP PolyBench Benchmark Suite

##State-of-the-Art Parallel Computing Benchmarks

A comprehensive benchmark suite implementing multiple parallelization strategies for PolyBench/C kernels using OpenMP. Designed for systematic performance evaluation, comparison with Julia implementations, and generation of publication-quality visualizations.

##  Benchmarks Included
### 1. **2MM** - Two Matrix Multiplications
- **Computation**: D = α·A·B·C + β·D
- **Characteristics**: Compute-intensive, regular patterns
- **Strategies**: 6 implementations including tiling, SIMD, task-based

### 2. **3MM** - Three Matrix Multiplications  
- **Computation**: E = A·B; F = C·D; G = E·F
- **Characteristics**: High arithmetic intensity, multiple stages
- **Strategies**: 6 implementations including pipeline, hierarchical

### 3. **Cholesky** - Matrix Decomposition
- **Computation**: A = L·L^T decomposition
- **Characteristics**: Dependencies, triangular operations
- **Strategies**: 6 implementations including recursive, left/right-looking

### 4. **Correlation** - Pearson Correlation Matrix
- **Computation**: Statistical correlation coefficients
- **Characteristics**: Reduction operations, memory-intensive
- **Strategies**: 5 implementations including row-wise, column-major

### 5. **Nussinov** - Dynamic Programming
- **Computation**: RNA secondary structure prediction
- **Characteristics**: Wavefront parallelism, irregular dependencies
- **Strategies**: 6 implementations including wavefront, pipeline, hybrid

##  Quick Start
### Prerequisites
```bash
# Required
gcc >= 7.0 (with OpenMP support)
make

# Optional
PAPI library (for hardware counters)
gnuplot (for visualization)
perf (for profiling)
```

### Basic Usage
```bash
# Clone and build
make all          # Build with default (STANDARD) size

# Run quick test
make test         # Run with SMALL size

# Run full benchmark suite
chmod +x run_benchmarks.sh
./run_benchmarks.sh

# Build with specific size
make medium       # or mini, small, large, xlarge
```

## Enhanced Performance Metrics

### Primary Metrics
1. **Time-based Performance**
   - Wall-clock execution time
   - CPU time utilization
   - Speedup: S(p) = T(1)/T(p)
   - Parallel efficiency: E(p) = S(p)/p × 100%
   - Strong/weak scaling analysis

2. **Resource Utilization**
   - CPU utilization percentage
   - Memory consumption (peak/average)
   - Memory bandwidth (GB/s)
   - Cache performance (with PAPI)

3. **Statistical Analysis**
   - Mean, standard deviation
   - 95% confidence intervals
   - Min/max times across iterations

4. **Advanced Metrics**
   - Arithmetic intensity (FLOPS/byte)
   - Load imbalance factor
   - Synchronization overhead
   - Amdahl's law serial fraction

##  Compilation Options

### Compiler Selection
```bash
make CC=gcc      # GNU Compiler (default)
make CC=icc      # Intel Compiler
make CC=clang    # LLVM/Clang
```

### Optimization Levels
```bash
make              # -O3 optimization (default)
make debug        # -O0 -g for debugging
make profile      # With profiling support
make o2           # -O2 optimization
```

### Problem Sizes
| Size | 2MM Dimensions | Cholesky N | Correlation M×N | Nussinov N |
|------|---------------|------------|-----------------|------------|
| MINI | 16×18×22×24 | 40 | 28×32 | 60 |
| SMALL | 40×50×70×80 | 120 | 80×100 | 180 |
| MEDIUM | 180×190×210×220 | 400 | 240×260 | 500 |
| LARGE | 800×900×1100×1200 | 2000 | 1200×1400 | 2500 |
| XLARGE | 1600×1800×2200×2400 | 4000 | 2600×3000 | 5500 |

##  Parallelization Strategies

### Common Strategies Across Benchmarks

1. **Sequential Baseline** - Reference implementation
2. **Basic Parallel** - Simple OpenMP parallelization
3. **Collapsed Loops** - Multi-dimensional parallelization
4. **Tiled/Blocked** - Cache optimization with blocking
5. **SIMD Vectorization** - Explicit vectorization directives
6. **Task-based** - Dynamic task decomposition with dependencies
7. **Pipeline** - Staged execution model
8. **Hybrid** - Combined coarse and fine-grained parallelism

### Strategy Selection Guidelines

- **Compute-bound** (2MM, 3MM): Use tiling + SIMD
- **Memory-bound** (Correlation): Focus on cache optimization
- **Dependency-heavy** (Cholesky): Task-based or recursive
- **Irregular** (Nussinov): Wavefront or hybrid approaches

##  Running Benchmarks

### Automated Benchmark Suite
```bash
# Default run (all benchmarks, multiple thread counts)
./run_benchmarks.sh

# Custom configuration
./run_benchmarks.sh -t "2 4 8 16" -s "small medium large"

# Specific benchmarks only
./run_benchmarks.sh -b "2mm cholesky" -t "1 2 4 8"

# Verbose output
./run_benchmarks.sh -v

# Dry run (see what would execute)
./run_benchmarks.sh -d
```

### Individual Benchmark Execution
```bash
# Set thread count
export OMP_NUM_THREADS=8

# Set thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Run specific benchmark
./benchmark_2mm
./benchmark_cholesky
```

### Environment Tuning
```bash
# For compute-bound kernels
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# For memory-bound kernels  
export OMP_PROC_BIND=spread
export OMP_PLACES=sockets

# Dynamic scheduling
export OMP_SCHEDULE="dynamic,1"
```

##  Result Analysis

### Output Files
```
results/
├── benchmark_results_TIMESTAMP.csv    # Raw data
├── benchmark_summary_TIMESTAMP.txt    # Human-readable summary
├── benchmark_results_TIMESTAMP.json   # JSON for processing
└── *_scaling_TIMESTAMP.png           # Performance plots
```

### CSV Format
```csv
Benchmark,Size,Threads,Avg_Time,Min_Time,Max_Time
2mm,medium,8,0.1234,0.1200,0.1300
```

### JSON Structure
```json
{
  "timestamp": "20231121_143022",
  "system": {
    "hostname": "node001",
    "cpu": "Intel Xeon Gold 6248",
    "cores": 40
  },
  "results": [
    {
      "benchmark": "2mm",
      "size": "medium",
      "threads": 8,
      "avg_time": 0.1234,
      "speedup": 6.5,
      "efficiency": 81.25
    }
  ]
}
```

## Advanced Analysis

### Profiling
```bash
# Build with profiling
make profile

# Run with perf
perf record -g ./benchmark_2mm
perf report

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > perf.svg
```

### Vectorization Analysis
```bash
# Check vectorization
make vec-report
cat vectorization.log

# Generate assembly
make asm
objdump -d benchmark_2mm.s | grep -A5 vmovapd
```

### PAPI Hardware Counters
```bash
# Build with PAPI support
make papi

# Counters collected:
# - L1/L2/L3 cache misses
# - TLB misses
# - Branch mispredictions
```

## Julia Comparison

The benchmark suite is designed for direct comparison with Julia implementations:

### Metrics for Comparison
1. **Execution Time** - Direct performance comparison
2. **GFLOPS** - Computational throughput
3. **Memory Efficiency** - Peak memory usage
4. **Scaling Behavior** - Thread efficiency curves
5. **Code Complexity** - Lines of code, implementation effort

### Export for Julia Processing
```julia
# Read CSV results in Julia
using CSV, DataFrames
results = CSV.read("results/benchmark_results.csv", DataFrame)

# Read JSON for detailed analysis
using JSON
data = JSON.parsefile("results/benchmark_results.json")
```

##  Visualization

### Automatic Plot Generation
The benchmark script automatically generates scaling plots using gnuplot:
- Thread scaling curves
- Speedup analysis
- Efficiency heatmaps

### Manual Visualization
```bash
# Generate custom plots
gnuplot plot_scaling.gnu

# Generate heatmap
python3 generate_heatmap.py results/benchmark_results.csv
```

## Troubleshooting

### Common Issues

1. **Poor Scaling**
   - Check thread affinity: `OMP_DISPLAY_ENV=TRUE ./benchmark_2mm`
   - Verify no CPU frequency scaling: `cpupower frequency-info`
   - Check for NUMA effects: `numactl --hardware`

2. **Incorrect Results**
   - Verify with debug build: `make debug`
   - Check for race conditions with thread sanitizer
   - Reduce optimization level: `make o2`

3. **Build Failures**
   - Verify OpenMP support: `gcc -fopenmp --version`
   - Check aligned allocation: requires POSIX 2008
   - For SIMD: ensure `-march=native` or specify architecture

##  Contributing
Contributions welcome! Areas of interest:
- Additional parallelization strategies
- GPU offloading with OpenMP 5.0
- MPI distributed implementations
- Additional PolyBench kernels
- Improved visualization tools

##  References

1. **PolyBench/C 4.2.1**: Pouchet et al., Ohio State University
2. **OpenMP Specification 5.0**: OpenMP ARB
3. **Intel Optimization Guide**: Intel Corporation
4. **"Patterns for Parallel Programming"**: Mattson, Sanders, Massingill

##  License

This benchmark suite is provided for educational and research purposes.
PolyBench is distributed under its own license terms.

##  Acknowledgments
- PolyBench/C authors for the benchmark kernels
- OpenMP community for parallelization patterns
- Project contributors and testers

---

**Author**: OpenMP Performance Engineering Team  
**Version**: 1.0  
**Last Updated**: November 2024

For questions or issues, please open an issue on the project repository.
