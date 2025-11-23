/**
 * 3mm.c: Three Matrix Multiplication Benchmark with Multiple Strategies
 * E = A*B; F = C*D; G = E*F
 * 
 * Strategies implemented:
 * 1. Sequential (baseline)
 * 2. Basic parallel
 * 3. Collapsed loops
 * 4. Tiled/Blocked
 * 5. SIMD vectorization
 * 6. Task-based with dependencies
 * 7. Pipeline parallel
 * 8. Hierarchical with prefetching
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "benchmark_metrics.h"

// Problem size definitions
#ifdef MINI
#define NI 16
#define NJ 18
#define NK 20
#define NL 22
#define NM 24
#elif defined(SMALL)
#define NI 40
#define NJ 50
#define NK 60
#define NL 70
#define NM 80
#elif defined(MEDIUM)
#define NI 180
#define NJ 190
#define NK 200
#define NL 210
#define NM 220
#elif defined(LARGE)
#define NI 800
#define NJ 900
#define NK 1000
#define NL 1100
#define NM 1200
#else // Default STANDARD
#define NI 100
#define NJ 120
#define NK 140
#define NL 160
#define NM 180
#endif

#define ALIGN_SIZE 64
#define TILE_SIZE 32

// Aligned memory allocation
static void* aligned_malloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) {
        fprintf(stderr, "Error: aligned allocation failed\n");
        exit(1);
    }
    memset(ptr, 0, size);
    return ptr;
}

// Initialize matrices
static void init_array(int ni, int nj, int nk, int nl, int nm,
                      double *A, double *B, double *C, double *D) {
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nk; j++)
            A[i*nk + j] = (double) ((i*j+1) % ni) / (5*ni);
    
    for (int i = 0; i < nk; i++)
        for (int j = 0; j < nj; j++)
            B[i*nj + j] = (double) ((i*(j+1)+2) % nj) / (5*nj);
    
    for (int i = 0; i < nj; i++)
        for (int j = 0; j < nm; j++)
            C[i*nm + j] = (double) (i*(j+3) % nl) / (5*nl);
    
    for (int i = 0; i < nm; i++)
        for (int j = 0; j < nl; j++)
            D[i*nl + j] = (double) ((i*(j+2)+2) % nk) / (5*nk);
}

// Verify results
static double verify_result(int ni, int nl, double *G_ref, double *G_test) {
    double max_error = 0.0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double error = fabs(G_ref[i*nl + j] - G_test[i*nl + j]);
            if (error > max_error) max_error = error;
        }
    }
    return max_error;
}

// Calculate FLOPS for 3mm
static long long calculate_flops(int ni, int nj, int nk, int nl, int nm) {
    // E = A*B: ni*nj*nk multiply-add operations
    // F = C*D: nj*nl*nm multiply-add operations
    // G = E*F: ni*nl*nj multiply-add operations
    // Each multiply-add = 2 operations
    return 2LL * (ni * nj * nk + nj * nl * nm + ni * nl * nj);
}

// Strategy 1: Sequential baseline
void kernel_3mm_sequential(int ni, int nj, int nk, int nl, int nm,
                          double *A, double *B, double *C, double *D,
                          double *E, double *F, double *G) {
    // E = A * B
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            E[i*nj + j] = 0.0;
            for (int k = 0; k < nk; k++) {
                E[i*nj + j] += A[i*nk + k] * B[k*nj + j];
            }
        }
    }
    
    // F = C * D
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nl; j++) {
            F[i*nl + j] = 0.0;
            for (int k = 0; k < nm; k++) {
                F[i*nl + j] += C[i*nm + k] * D[k*nl + j];
            }
        }
    }
    
    // G = E * F
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            G[i*nl + j] = 0.0;
            for (int k = 0; k < nj; k++) {
                G[i*nl + j] += E[i*nj + k] * F[k*nl + j];
            }
        }
    }
}

// Strategy 2: Basic parallel
void kernel_3mm_basic_parallel(int ni, int nj, int nk, int nl, int nm,
                               double *A, double *B, double *C, double *D,
                               double *E, double *F, double *G) {
    // E = A * B
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            for (int k = 0; k < nk; k++) {
                sum += A[i*nk + k] * B[k*nj + j];
            }
            E[i*nj + j] = sum;
        }
    }
    
    // F = C * D
    #pragma omp parallel for
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = 0.0;
            for (int k = 0; k < nm; k++) {
                sum += C[i*nm + k] * D[k*nl + j];
            }
            F[i*nl + j] = sum;
        }
    }
    
    // G = E * F
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = 0.0;
            for (int k = 0; k < nj; k++) {
                sum += E[i*nj + k] * F[k*nl + j];
            }
            G[i*nl + j] = sum;
        }
    }
}

// Strategy 3: Collapsed loops
void kernel_3mm_collapsed(int ni, int nj, int nk, int nl, int nm,
                         double *A, double *B, double *C, double *D,
                         double *E, double *F, double *G) {
    // E = A * B
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            for (int k = 0; k < nk; k++) {
                sum += A[i*nk + k] * B[k*nj + j];
            }
            E[i*nj + j] = sum;
        }
    }
    
    // F = C * D
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = 0.0;
            for (int k = 0; k < nm; k++) {
                sum += C[i*nm + k] * D[k*nl + j];
            }
            F[i*nl + j] = sum;
        }
    }
    
    // G = E * F
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = 0.0;
            for (int k = 0; k < nj; k++) {
                sum += E[i*nj + k] * F[k*nl + j];
            }
            G[i*nl + j] = sum;
        }
    }
}

// Strategy 4: Tiled/Blocked
void kernel_3mm_tiled(int ni, int nj, int nk, int nl, int nm,
                     double *A, double *B, double *C, double *D,
                     double *E, double *F, double *G) {
    const int tile = TILE_SIZE;
    
    // Initialize E
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
            E[i*nj + j] = 0.0;
    
    // E = A * B with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += tile) {
        for (int jj = 0; jj < nj; jj += tile) {
            for (int kk = 0; kk < nk; kk += tile) {
                int i_end = (ii + tile < ni) ? ii + tile : ni;
                int j_end = (jj + tile < nj) ? jj + tile : nj;
                int k_end = (kk + tile < nk) ? kk + tile : nk;
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double aik = A[i*nk + k];
                        #pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            E[i*nj + j] += aik * B[k*nj + j];
                        }
                    }
                }
            }
        }
    }
    
    // Initialize F
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nj; i++)
        for (int j = 0; j < nl; j++)
            F[i*nl + j] = 0.0;
    
    // F = C * D with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < nj; ii += tile) {
        for (int jj = 0; jj < nl; jj += tile) {
            for (int kk = 0; kk < nm; kk += tile) {
                int i_end = (ii + tile < nj) ? ii + tile : nj;
                int j_end = (jj + tile < nl) ? jj + tile : nl;
                int k_end = (kk + tile < nm) ? kk + tile : nm;
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double cik = C[i*nm + k];
                        #pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            F[i*nl + j] += cik * D[k*nl + j];
                        }
                    }
                }
            }
        }
    }
    
    // G = E * F with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += tile) {
        for (int jj = 0; jj < nl; jj += tile) {
            int i_end = (ii + tile < ni) ? ii + tile : ni;
            int j_end = (jj + tile < nl) ? jj + tile : nl;
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < nj; k++) {
                        sum += E[i*nj + k] * F[k*nl + j];
                    }
                    G[i*nl + j] = sum;
                }
            }
        }
    }
}

// Strategy 5: SIMD vectorization
void kernel_3mm_simd(int ni, int nj, int nk, int nl, int nm,
                    double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ C, double *__restrict__ D,
                    double *__restrict__ E, double *__restrict__ F,
                    double *__restrict__ G) {
    // Assume aligned pointers for compiler optimization
    A = __builtin_assume_aligned(A, ALIGN_SIZE);
    B = __builtin_assume_aligned(B, ALIGN_SIZE);
    C = __builtin_assume_aligned(C, ALIGN_SIZE);
    D = __builtin_assume_aligned(D, ALIGN_SIZE);
    E = __builtin_assume_aligned(E, ALIGN_SIZE);
    F = __builtin_assume_aligned(F, ALIGN_SIZE);
    G = __builtin_assume_aligned(G, ALIGN_SIZE);
    
    // E = A * B with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        #pragma omp simd aligned(E:ALIGN_SIZE)
        for (int j = 0; j < nj; j++) {
            E[i*nj + j] = 0.0;
        }
        
        for (int k = 0; k < nk; k++) {
            double aik = A[i*nk + k];
            #pragma omp simd aligned(B,E:ALIGN_SIZE)
            for (int j = 0; j < nj; j++) {
                E[i*nj + j] += aik * B[k*nj + j];
            }
        }
    }
    
    // F = C * D with SIMD
    #pragma omp parallel for
    for (int i = 0; i < nj; i++) {
        #pragma omp simd aligned(F:ALIGN_SIZE)
        for (int j = 0; j < nl; j++) {
            F[i*nl + j] = 0.0;
        }
        
        for (int k = 0; k < nm; k++) {
            double cik = C[i*nm + k];
            #pragma omp simd aligned(D,F:ALIGN_SIZE)
            for (int j = 0; j < nl; j++) {
                F[i*nl + j] += cik * D[k*nl + j];
            }
        }
    }
    
    // G = E * F with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum) aligned(E,F:ALIGN_SIZE)
            for (int k = 0; k < nj; k++) {
                sum += E[i*nj + k] * F[k*nl + j];
            }
            G[i*nl + j] = sum;
        }
    }
}

// Strategy 6: Task-based with dependencies
void kernel_3mm_tasks(int ni, int nj, int nk, int nl, int nm,
                     double *A, double *B, double *C, double *D,
                     double *E, double *F, double *G) {
    const int chunk = 32;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // E = A * B with tasks
            for (int i = 0; i < ni; i += chunk) {
                for (int j = 0; j < nj; j += chunk) {
                    #pragma omp task depend(out:E[i*nj+j:chunk*chunk])
                    {
                        int i_end = (i + chunk < ni) ? i + chunk : ni;
                        int j_end = (j + chunk < nj) ? j + chunk : nj;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = 0.0;
                                for (int k = 0; k < nk; k++) {
                                    sum += A[ii*nk + k] * B[k*nj + jj];
                                }
                                E[ii*nj + jj] = sum;
                            }
                        }
                    }
                }
            }
            
            // F = C * D with tasks (can run in parallel with E)
            for (int i = 0; i < nj; i += chunk) {
                for (int j = 0; j < nl; j += chunk) {
                    #pragma omp task depend(out:F[i*nl+j:chunk*chunk])
                    {
                        int i_end = (i + chunk < nj) ? i + chunk : nj;
                        int j_end = (j + chunk < nl) ? j + chunk : nl;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = 0.0;
                                for (int k = 0; k < nm; k++) {
                                    sum += C[ii*nm + k] * D[k*nl + jj];
                                }
                                F[ii*nl + jj] = sum;
                            }
                        }
                    }
                }
            }
            
            // G = E * F with dependencies
            for (int i = 0; i < ni; i += chunk) {
                for (int j = 0; j < nl; j += chunk) {
                    #pragma omp task depend(in:E[i*nj:chunk*nj],F[0:nj*nl]) \
                                     depend(out:G[i*nl+j:chunk*chunk])
                    {
                        int i_end = (i + chunk < ni) ? i + chunk : ni;
                        int j_end = (j + chunk < nl) ? j + chunk : nl;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = 0.0;
                                for (int k = 0; k < nj; k++) {
                                    sum += E[ii*nj + k] * F[k*nl + jj];
                                }
                                G[ii*nl + jj] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Strategy 7: Pipeline parallel
void kernel_3mm_pipeline(int ni, int nj, int nk, int nl, int nm,
                        double *A, double *B, double *C, double *D,
                        double *E, double *F, double *G) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Stage 1: Compute E (subset of rows)
        int e_chunk = (ni + num_threads - 1) / num_threads;
        int e_start = tid * e_chunk;
        int e_end = (e_start + e_chunk < ni) ? e_start + e_chunk : ni;
        
        for (int i = e_start; i < e_end; i++) {
            for (int j = 0; j < nj; j++) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < nk; k++) {
                    sum += A[i*nk + k] * B[k*nj + j];
                }
                E[i*nj + j] = sum;
            }
        }
        
        // Stage 2: Compute F (subset of rows)
        int f_chunk = (nj + num_threads - 1) / num_threads;
        int f_start = tid * f_chunk;
        int f_end = (f_start + f_chunk < nj) ? f_start + f_chunk : nj;
        
        for (int i = f_start; i < f_end; i++) {
            for (int j = 0; j < nl; j++) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < nm; k++) {
                    sum += C[i*nm + k] * D[k*nl + j];
                }
                F[i*nl + j] = sum;
            }
        }
        
        #pragma omp barrier
        
        // Stage 3: Compute G (subset of rows)
        int g_chunk = (ni + num_threads - 1) / num_threads;
        int g_start = tid * g_chunk;
        int g_end = (g_start + g_chunk < ni) ? g_start + g_chunk : ni;
        
        for (int i = g_start; i < g_end; i++) {
            for (int j = 0; j < nl; j++) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < nj; k++) {
                    sum += E[i*nj + k] * F[k*nl + j];
                }
                G[i*nl + j] = sum;
            }
        }
    }
}

// Main benchmark driver
int main(int argc, char** argv) {
    // Allocate aligned memory
    double *A = (double*)aligned_malloc(NI * NK * sizeof(double));
    double *B = (double*)aligned_malloc(NK * NJ * sizeof(double));
    double *C = (double*)aligned_malloc(NJ * NM * sizeof(double));
    double *D = (double*)aligned_malloc(NM * NL * sizeof(double));
    double *E = (double*)aligned_malloc(NI * NJ * sizeof(double));
    double *F = (double*)aligned_malloc(NJ * NL * sizeof(double));
    double *G_ref = (double*)aligned_malloc(NI * NL * sizeof(double));
    double *G = (double*)aligned_malloc(NI * NL * sizeof(double));
    
    // Initialize matrices
    init_array(NI, NJ, NK, NL, NM, A, B, C, D);
    
    // Calculate FLOPS
    long long total_flops = calculate_flops(NI, NJ, NK, NL, NM);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    printf("\n=== Running 3MM Benchmark ===\n");
    printf("Problem size: NI=%d, NJ=%d, NK=%d, NL=%d, NM=%d\n", 
           NI, NJ, NK, NL, NM);
    printf("Total FLOPS: %lld\n", total_flops);
    printf("Memory footprint: %.2f MB\n\n",
           (NI*NK + NK*NJ + NJ*NM + NM*NL + NI*NJ + NJ*NL + 2*NI*NL) * 
           sizeof(double) / (1024.0*1024.0));
    
    // Sequential baseline
    double start = omp_get_wtime();
    kernel_3mm_sequential(NI, NJ, NK, NL, NM, A, B, C, D, E, F, G_ref);
    double serial_time = omp_get_wtime() - start;
    printf("Sequential: %.4f seconds (%.2f GFLOPS)\n\n",
           serial_time, total_flops / (serial_time * 1e9));
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    int num_thread_configs = 4;
    
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "Strategy", "Threads", "Time (s)", "Speedup", "Efficiency", "GFLOPS");
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "--------", "-------", "--------", "-------", "----------", "------");
    
    // Define strategies
    typedef void (*strategy_func)(int, int, int, int, int,
                                 double*, double*, double*, double*,
                                 double*, double*, double*);
    
    struct {
        const char* name;
        strategy_func func;
    } strategies[] = {
        {"Basic Parallel", kernel_3mm_basic_parallel},
        {"Collapsed", kernel_3mm_collapsed},
        {"Tiled", kernel_3mm_tiled},
        {"SIMD", kernel_3mm_simd},
        {"Task-based", kernel_3mm_tasks},
        {"Pipeline", kernel_3mm_pipeline}
    };
    
    // Test each strategy
    for (int s = 0; s < 6; s++) {
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);
            
            // Time the strategy
            double times[MEASUREMENT_ITERATIONS];
            for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
                memset(E, 0, NI * NJ * sizeof(double));
                memset(F, 0, NJ * NL * sizeof(double));
                memset(G, 0, NI * NL * sizeof(double));
                
                start = omp_get_wtime();
                strategies[s].func(NI, NJ, NK, NL, NM, A, B, C, D, E, F, G);
                times[iter] = omp_get_wtime() - start;
            }
            
            // Calculate average time
            double avg_time = 0.0;
            for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
                avg_time += times[i];
            }
            avg_time /= MEASUREMENT_ITERATIONS;
            
            // Verify correctness
            memset(E, 0, NI * NJ * sizeof(double));
            memset(F, 0, NJ * NL * sizeof(double));
            memset(G, 0, NI * NL * sizeof(double));
            strategies[s].func(NI, NJ, NK, NL, NM, A, B, C, D, E, F, G);
            double error = verify_result(NI, NL, G_ref, G);
            
            // Calculate metrics
            double speedup = serial_time / avg_time;
            double efficiency = speedup / num_threads * 100.0;
            double gflops = total_flops / (avg_time * 1e9);
            
            printf("%-25s %-10d %-12.4f %-12.2f %-12.1f%% %-10.2f",
                   strategies[s].name, num_threads, avg_time, speedup, efficiency, gflops);
            
            if (error > 1e-10) {
                printf(" [ERROR: %.2e]", error);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // Free memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    free(G_ref);
    free(G);
    
    return 0;
}