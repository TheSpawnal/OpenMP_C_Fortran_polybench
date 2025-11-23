/**
 * missing_strategies.c: Implementations of the strategies mentioned but missing
 * from the original 2mm.c and 3mm.c benchmarks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

// ============================================================================
// MISSING STRATEGY 1: Hierarchical (Nested Parallelism) for 2MM
// ============================================================================

/**
 * Hierarchical parallelization using nested OpenMP
 * - Outer level: Distributes major computation blocks
 * - Inner level: Fine-grained parallelism within blocks
 * 
 * Benefits:
 * - Better for NUMA systems (not applicable to your system)
 * - Can balance heterogeneous workloads
 * - Demonstrates nested parallelism concepts
 */
void kernel_2mm_hierarchical(int ni, int nj, int nk, int nl,
                             double alpha, double beta,
                             double *A, double *B, double *C, double *D,
                             double *tmp) {
    // Enable nested parallelism
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    
    // First multiplication: tmp = alpha * A * B
    // Use 2 teams at outer level, each with multiple threads
    #pragma omp parallel num_threads(2)
    {
        int team_id = omp_get_thread_num();
        
        // Divide work between teams
        int rows_per_team = (ni + 1) / 2;
        int start_row = team_id * rows_per_team;
        int end_row = (team_id == 0) ? rows_per_team : ni;
        
        // Each team uses inner parallelism
        #pragma omp parallel for collapse(2) num_threads(4)
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < nj; j++) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < nk; k++) {
                    sum += alpha * A[i*nk + k] * B[k*nj + j];
                }
                tmp[i*nj + j] = sum;
            }
        }
    }
    
    // Second multiplication: D = tmp * C + beta * D
    // Use full parallelism (all threads)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i*nl + j];
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp[i*nj + k] * C[k*nl + j];
            }
            D[i*nl + j] = sum;
        }
    }
}

// ============================================================================
// MISSING STRATEGY 2: Cache-optimized with Prefetching for 2MM
// ============================================================================

/**
 * Advanced cache optimization with software prefetching
 * 
 * Techniques used:
 * - Cache-line aware tiling
 * - Software prefetching with __builtin_prefetch
 * - Loop reordering for better spatial locality
 * - Minimized cache line conflicts
 */
void kernel_2mm_prefetch(int ni, int nj, int nk, int nl,
                         double alpha, double beta,
                         double *A, double *B, double *C, double *D,
                         double *tmp) {
    const int TILE = 32;  // L1 cache-friendly
    const int PREFETCH_DISTANCE = 8;  // Tune based on cache latency
    
    // First multiplication with aggressive prefetching
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += TILE) {
        for (int jj = 0; jj < nj; jj += TILE) {
            // Initialize tile
            int i_end = (ii + TILE < ni) ? ii + TILE : ni;
            int j_end = (jj + TILE < nj) ? jj + TILE : nj;
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    tmp[i*nj + j] = 0.0;
                }
            }
            
            // Compute tile with prefetching
            for (int kk = 0; kk < nk; kk += TILE) {
                int k_end = (kk + TILE < nk) ? kk + TILE : nk;
                
                for (int i = ii; i < i_end; i++) {
                    // Prefetch next rows of A
                    if (i + PREFETCH_DISTANCE < i_end) {
                        __builtin_prefetch(&A[(i+PREFETCH_DISTANCE)*nk + kk], 0, 3);
                    }
                    
                    for (int k = kk; k < k_end; k++) {
                        double aik = alpha * A[i*nk + k];
                        
                        // Prefetch B rows ahead
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
    
    // Second multiplication with prefetching
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) {
        // Prefetch ahead in tmp and C
        if (i + PREFETCH_DISTANCE < ni) {
            __builtin_prefetch(&tmp[(i+PREFETCH_DISTANCE)*nj], 0, 3);
        }
        
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i*nl + j];
            
            // Prefetch next column of C
            if (j + PREFETCH_DISTANCE < nl) {
                __builtin_prefetch(&C[0*nl + (j+PREFETCH_DISTANCE)], 0, 3);
            }
            
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp[i*nj + k] * C[k*nl + j];
            }
            D[i*nl + j] = sum;
        }
    }
}

// ============================================================================
// MISSING STRATEGY 3: Hierarchical with Prefetching for 3MM
// ============================================================================

/**
 * Combines hierarchical parallelism with cache optimization
 * Demonstrates advanced optimization for the 3MM kernel
 */
void kernel_3mm_hierarchical_prefetch(int ni, int nj, int nk, int nl, int nm,
                                     double *A, double *B, double *C, double *D,
                                     double *E, double *F, double *G) {
    const int TILE = 32;
    const int PREFETCH_DIST = 8;
    
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    
    // Parallel computation of E and F (independent!)
    #pragma omp parallel num_threads(2)
    {
        int team = omp_get_thread_num();
        
        if (team == 0) {
            // Team 0: Compute E = A * B
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < ni; i++) {
                // Prefetch ahead
                if (i + PREFETCH_DIST < ni) {
                    __builtin_prefetch(&A[(i+PREFETCH_DIST)*nk], 0, 3);
                }
                
                for (int j = 0; j < nj; j++) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < nk; k++) {
                        sum += A[i*nk + k] * B[k*nj + j];
                    }
                    E[i*nj + j] = sum;
                }
            }
        } else {
            // Team 1: Compute F = C * D
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < nj; i++) {
                // Prefetch ahead
                if (i + PREFETCH_DIST < nj) {
                    __builtin_prefetch(&C[(i+PREFETCH_DIST)*nm], 0, 3);
                }
                
                for (int j = 0; j < nl; j++) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < nm; k++) {
                        sum += C[i*nm + k] * D[k*nl + j];
                    }
                    F[i*nl + j] = sum;
                }
            }
        }
    }
    
    // G = E * F (after barrier, both E and F are ready)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) {
        // Prefetch E rows
        if (i + PREFETCH_DIST < ni) {
            __builtin_prefetch(&E[(i+PREFETCH_DIST)*nj], 0, 3);
        }
        
        for (int j = 0; j < nl; j++) {
            double sum = 0.0;
            
            // Prefetch F columns
            if (j + PREFETCH_DIST < nl) {
                __builtin_prefetch(&F[0*nl + (j+PREFETCH_DIST)], 0, 3);
            }
            
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += E[i*nj + k] * F[k*nl + j];
            }
            G[i*nl + j] = sum;
        }
    }
}

// ============================================================================
// BONUS: Wavefront Parallelization for 2MM
// ============================================================================

/**
 * Wavefront/diagonal scheduling
 * Exploits parallelism along diagonals in computation
 * Useful when there are loop-carried dependencies
 * 
 * Note: 2MM doesn't have dependencies, but this demonstrates the technique
 */
void kernel_2mm_wavefront(int ni, int nj, int nk, int nl,
                         double alpha, double beta,
                         double *A, double *B, double *C, double *D,
                         double *tmp) {
    // First multiplication - standard parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            for (int k = 0; k < nk; k++) {
                sum += alpha * A[i*nk + k] * B[k*nj + j];
            }
            tmp[i*nj + j] = sum;
        }
    }
    
    // Second multiplication - wavefront style (for demonstration)
    // Process elements along diagonals: i+j = constant
    int max_diagonal = ni + nl - 1;
    
    for (int diag = 0; diag < max_diagonal; diag++) {
        // All points on this diagonal can be computed in parallel
        #pragma omp parallel for
        for (int i = 0; i < ni; i++) {
            int j = diag - i;
            if (j >= 0 && j < nl) {
                double sum = beta * D[i*nl + j];
                for (int k = 0; k < nj; k++) {
                    sum += tmp[i*nj + k] * C[k*nl + j];
                }
                D[i*nl + j] = sum;
            }
        }
    }
}

// ============================================================================
// BONUS: Dynamic Load Balancing Strategy
// ============================================================================

/**
 * Uses OpenMP's guided scheduling for automatic load balancing
 * Particularly useful when work per iteration varies
 */
void kernel_2mm_dynamic(int ni, int nj, int nk, int nl,
                       double alpha, double beta,
                       double *A, double *B, double *C, double *D,
                       double *tmp) {
    // First multiplication with guided scheduling
    // Larger chunks initially, smaller chunks as work completes
    #pragma omp parallel for schedule(guided, 16)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nk; k++) {
                sum += alpha * A[i*nk + k] * B[k*nj + j];
            }
            tmp[i*nj + j] = sum;
        }
    }
    
    // Second multiplication with dynamic scheduling
    // Work stealing for better load balance
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i*nl + j];
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp[i*nj + k] * C[k*nl + j];
            }
            D[i*nl + j] = sum;
        }
    }
}

// ============================================================================
// UTILITY: Strategy Selection Heuristic
// ============================================================================

/**
 * Automatically selects best strategy based on problem characteristics
 */
typedef enum {
    STRAT_BASIC,
    STRAT_COLLAPSED,
    STRAT_TILED,
    STRAT_SIMD,
    STRAT_TASK,
    STRAT_HIERARCHICAL,
    STRAT_PREFETCH,
    STRAT_DYNAMIC
} StrategyType;

StrategyType select_best_strategy(int ni, int nj, int nk, int nl, 
                                   int num_threads, int cache_size_kb) {
    long long problem_size = (long long)ni * nj * nk;
    
    // Very small problem - basic parallel
    if (problem_size < 1000000) {
        return STRAT_BASIC;
    }
    
    // Few rows compared to threads - use collapse
    if (ni < num_threads * 2) {
        return STRAT_COLLAPSED;
    }
    
    // Large problem that fits in cache - use tiling
    size_t working_set = (ni * nk + nk * nj + ni * nj) * sizeof(double);
    if (working_set < cache_size_kb * 1024 / 2) {
        return STRAT_TILED;
    }
    
    // Very large problem - use prefetching
    if (problem_size > 100000000) {
        return STRAT_PREFETCH;
    }
    
    // Default to SIMD
    return STRAT_SIMD;
}

// ============================================================================
// TESTING HARNESS
// ============================================================================

/**
 * Example usage and testing function
 */
void test_missing_strategies() {
    const int NI = 100, NJ = 120, NK = 140, NL = 160;
    const double ALPHA = 1.5, BETA = 1.2;
    
    // Allocate matrices
    double *A = (double*)malloc(NI * NK * sizeof(double));
    double *B = (double*)malloc(NK * NJ * sizeof(double));
    double *C = (double*)malloc(NJ * NL * sizeof(double));
    double *D = (double*)malloc(NI * NL * sizeof(double));
    double *tmp = (double*)malloc(NI * NJ * sizeof(double));
    
    // Initialize (simple initialization for testing)
    for (int i = 0; i < NI * NK; i++) A[i] = (double)(i % 10) / 10.0;
    for (int i = 0; i < NK * NJ; i++) B[i] = (double)(i % 10) / 10.0;
    for (int i = 0; i < NJ * NL; i++) C[i] = (double)(i % 10) / 10.0;
    for (int i = 0; i < NI * NL; i++) D[i] = (double)(i % 10) / 10.0;
    
    printf("Testing missing strategies...\n");
    
    // Test hierarchical
    double start = omp_get_wtime();
    kernel_2mm_hierarchical(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
    double time_hier = omp_get_wtime() - start;
    printf("Hierarchical: %.4f seconds\n", time_hier);
    
    // Reset D
    for (int i = 0; i < NI * NL; i++) D[i] = (double)(i % 10) / 10.0;
    
    // Test prefetch
    start = omp_get_wtime();
    kernel_2mm_prefetch(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
    double time_prefetch = omp_get_wtime() - start;
    printf("Prefetch: %.4f seconds\n", time_prefetch);
    
    // Test wavefront
    for (int i = 0; i < NI * NL; i++) D[i] = (double)(i % 10) / 10.0;
    start = omp_get_wtime();
    kernel_2mm_wavefront(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
    double time_wave = omp_get_wtime() - start;
    printf("Wavefront: %.4f seconds\n", time_wave);
    
    // Test dynamic
    for (int i = 0; i < NI * NL; i++) D[i] = (double)(i % 10) / 10.0;
    start = omp_get_wtime();
    kernel_2mm_dynamic(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
    double time_dyn = omp_get_wtime() - start;
    printf("Dynamic: %.4f seconds\n", time_dyn);
    
    // Test strategy selection
    StrategyType recommended = select_best_strategy(NI, NJ, NK, NL, 
                                                    omp_get_max_threads(), 
                                                    256);  // 256KB L2
    printf("\nRecommended strategy: %d\n", recommended);
    
    // Cleanup
    free(A); free(B); free(C); free(D); free(tmp);
}

/**
 * To compile this file:
 * 
 * gcc -fopenmp -O3 -march=native missing_strategies.c -o test_missing -lm
 * 
 * Then run:
 * ./test_missing
 * 
 * To integrate into your benchmarks, copy the function implementations
 * into benchmark_2mm.c or benchmark_3mm.c and add them to your
 * strategies array.
 */