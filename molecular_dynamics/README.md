# Parallel Molecular Dynamics Simulation

## Overview

This project implements a molecular dynamics simulation of melting solid argon. The computation is dominated by force pair calculations in the `forces` subroutine, which has been parallelized using OpenMP.

## Project Structure

- `main.c` - Main simulation loop and initialization
- `forces.c` - Force calculation routine (parallelized)
- `domove.c` - Particle movement with periodic boundary conditions
- `mkekin.c` - Kinetic energy calculation
- `velavg.c` - Average velocity computation
- `mxwell.c` - Maxwell distribution sampling for velocities
- `fcc.c` - FCC lattice generation
- `dfill.c` - Array initialization
- `dscal.c` - Array scaling
- `prnout.c` - Output printing
- `random.c/h` - Random number generator

## Parallelization Strategy(cf MolDyn3/)

### 1. Current Implementation

The code uses OpenMP with the following structure:

#### In `forces.c`:
```c
#pragma omp single
{
    vir = 0.0;
    epot = 0.0;
}

#pragma omp for reduction(+:epot,vir) schedule(static,32)
for (i=0; i<npart*3; i+=3) {
    // Force calculations
}

#pragma omp for
for (i = 0; i < npart*3; i++) {
    // Accumulate per-thread forces
}
```

#### In `main.c`:
```c
#pragma omp parallel default(shared) private(move)
{
    for (move=1; move<=movemx; move++) {
        #pragma omp single
        {
            domove(3*npart, x, vh, f, side);
        }
        
        forces(npart, x, f, side, rcoff, ftemp, nthreads);
        
        #pragma omp single
        {
            ekin=mkekin(npart, f, vh, hsq2, hsq);
            vel=velavg(npart, vh, vaver, h);
            // ... scaling and output
        }
    }
}
```

### 2. Variable Classification

**SHARED Variables:**
- `x[]` - Particle positions
- `f[]` - Forces
- `vh[]` - Velocities
- `side`, `rcoff` - Simulation parameters
- `ftemp` - Temporary force array
- `npart` - Number of particles
- `epot`, `vir` - Energy and virial

**PRIVATE Variables:**
- `i`, `j` - Loop indices
- `myid` - Thread ID
- `fxi`, `fyi`, `fzi` - Force components
- `xx`, `yy`, `zz` - Distance components
- `rd`, `rrd`, `rrd3`, `rrd4`, `r148` - Distance calculations
- `id` - Thread index in accumulation loop
- `move` - Time step counter (in main)

**REDUCTION Variables:**
- `epot` - Potential energy
- `vir` - Virial

### 3. Avoiding Atomics Bottleneck

The code uses a **temporary force array** approach to avoid atomic operations:

```c
double **ftemp;  // ftemp[nthreads][npart*3]
```

**How it works:**
1. Each thread writes to its own temporary array: `ftemp[myid][j]`
2. No race conditions during force calculation (most expensive part)
3. After force calculation, all threads accumulate their results into `f[]`
4. This eliminates the need for atomic operations during the critical computation

**Benefits:**
- No atomic overhead during force calculations
- Better cache performance
- Scales well with thread count

## Scheduling Experiments

### Understanding Scheduling

The `schedule` clause controls how loop iterations are distributed among threads.

### Current Schedule: `schedule(static,32)`

**How it works:**
- Divides iterations into fixed chunks of 32 before execution
- Assigns chunks to threads in round-robin fashion
- Thread 0: iterations 0-31, Thread 1: iterations 32-63, etc.

**Pros:**
- Very low overhead
- Predictable performance

**Cons:**
- Load imbalance in this application
- Particle 0 computes forces with ALL other particles
- Particle (npart-1) has minimal work

### Alternative Schedules to Try

#### 1. Dynamic Scheduling
```c
#pragma omp for reduction(+:epot,vir) schedule(dynamic,32)
```
- Chunks assigned at runtime as threads finish work
- Better load balancing for irregular workloads
- Higher overhead than static

**When to use:** When work per iteration varies significantly

#### 2. Guided Scheduling
```c
#pragma omp for reduction(+:epot,vir) schedule(guided)
```
- Chunk size decreases over time
- Starts with large chunks, ends with small ones
- Good compromise between overhead and load balance

**When to use:** For gradual load imbalance (like this application)

#### 3. Smaller Static Chunks
```c
#pragma omp for reduction(+:epot,vir) schedule(static,8)
```
- Better distribution of expensive and cheap iterations
- Still low overhead

**When to use:** When static scheduling is desired but needs better distribution

#### 4. Runtime Scheduling
```c
#pragma omp for reduction(+:epot,vir) schedule(runtime)
```
- Schedule determined by `OMP_SCHEDULE` environment variable
- Useful for experimentation without recompilation

**Usage:**
```bash
export OMP_SCHEDULE="dynamic,16"
./md_simulation
```

#### 5. Auto Scheduling
```c
#pragma omp for reduction(+:epot,vir) schedule(auto)
```
- Compiler/runtime decides the best schedule
- Implementation-dependent

### Recommended Experiments

1. **Baseline:** `schedule(static,32)` (current)
2. **Test:** `schedule(dynamic,32)`
3. **Test:** `schedule(dynamic,16)`
4. **Test:** `schedule(guided)`
5. **Test:** `schedule(static,8)`

Measure execution time for each and compare performance.

## Alternative Approach: Locks

Instead of temporary arrays, locks can be used (though generally less efficient):

### Implementation

```c
omp_lock_t locks[npart];

// Initialize locks
for(i=0; i<npart; i++) 
    omp_init_lock(&locks[i]);

// In force calculation
omp_set_lock(&locks[j/3]);
f[j]   -= xx*r148;
f[j+1] -= yy*r148;
f[j+2] -= zz*r148;
omp_unset_lock(&locks[j/3]);

// Cleanup
for(i=0; i<npart; i++)
    omp_destroy_lock(&locks[i]);
```

### Optimization Considerations

**Number of particles per lock:**
- **Too few locks (many particles per lock):** High contention, threads wait often
- **Too many locks (one per particle):** High memory overhead, cache issues
- **Optimal:** 10-100 particles per lock balances overhead and contention

**Trade-offs:**
- Locks have higher overhead than temporary arrays
- But use less memory for many threads
- Temporary array approach is generally preferred

## Compilation

```bash
gcc -fopenmp -O3 -o md_simulation *.c -lm
```

## Execution

```bash
export OMP_NUM_THREADS=4
./md_simulation
```

## Performance Analysis

### Key Metrics

1. **Total execution time**
2. **Load balance** (check if all threads finish at similar times)
3. **Speedup** = Time(serial) / Time(parallel)
4. **Efficiency** = Speedup / Number_of_threads

### Expected Behavior

- **Load imbalance** with static scheduling due to N² force calculation pattern
- **Better performance** with dynamic or guided scheduling
- **Optimal thread count** depends on system (typically number of physical cores)

## Design Benefits

### Parallel Region Outside Loop

Moving the parallel region to encompass the entire time-step loop avoids:
- Repeated thread creation/destruction overhead
- Better cache locality
- Reduced synchronization overhead

### Single Thread Sections

Functions like `domove`, `mkekin`, `velavg` are protected with `#pragma omp single` because:
- They're not parallelized themselves
- They operate on shared data structures
- Only one thread should execute them to avoid race conditions

## Assignment Tasks Summary

1. **Parallelize forces routine** - Already done with reduction variables
2. **Move parallel region to main.c** - Already encompassing iteration loop
3.  **Avoid atomics bottleneck** - Using temporary force array `ftemp`
4.  **Experiment with schedules** - Try dynamic, guided, different chunk sizes
5.  **Analyze performance** - Compare timing results

## Further Optimization Ideas

1. **SIMD vectorization** - Use compiler auto-vectorization or intrinsics
2. **Cache optimization** - Improve data layout for better cache utilization
3. **Cutoff optimization** - Use neighbor lists to reduce force calculations
4. **Hybrid parallelization** - Combine OpenMP with MPI for distributed systems

## References

- OpenMP API Specification: https://www.openmp.org/specifications/
- Molecular Dynamics algorithms: Frenkel & Smit, "Understanding Molecular Simulation"
