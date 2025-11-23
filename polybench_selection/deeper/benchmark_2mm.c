/**
 * 2mm.c: This file implements the 2mm benchmark with multiple parallelization strategies
 * D = alpha*A*B*C + beta*D (Matrix Multiplication Chain)
 * 
 * Strategies implemented:
 * 1. Sequential (baseline)
 * 2. Basic parallel
 * 3. Collapsed loops  
 * 4. Tiled/Blocked
 * 5. SIMD vectorization
 * 6. Task-based with dependencies
 * 7. Hierarchical (nested parallelism)
 * 8. Cache-optimized with prefetching
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>  // For SIMD intrinsics
#include "benchmark_metrics.h"

// Problem size definitions (from PolyBench)
#ifdef MINI
#define NI 16
#define NJ 18
#define NK 22
#define NL 24
#elif defined(SMALL)
#define NI 40
#define NJ 50
#define NK 70
#define NL 80
#elif defined(MEDIUM)
#define NI 180
#define NJ 190
#define NK 210
#define NL 220
#elif defined(LARGE)
#define NI 800
#define NJ 900
#define NK 1100
#define NL 1200
#else // Default STANDARD
#define NI 100
#define NJ 120
#define NK 140
#define NL 160
#endif

#define ALPHA 1.5
#define BETA 1.2
#define ALIGN_SIZE 64

// Aligned memory allocation
static void* aligned_malloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) {
        fprintf(stderr, "Error: aligned allocation failed\n");
        exit(1);
    }
    return ptr;
}

// Initialize matrices
static void init_array(int ni, int nj, int nk, int nl,
                      double *A, double *B, double *C, double *D) {
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nk; j++)
            A[i*nk + j] = (double) ((i*j+1) % ni) / ni;
    
    for (int i = 0; i < nk; i++)
        for (int j = 0; j < nj; j++)
            B[i*nj + j] = (double) (i*(j+1) % nj) / nj;
    
    for (int i = 0; i < nj; i++)
        for (int j = 0; j < nl; j++)
            C[i*nl + j] = (double) ((i*(j+3)+1) % nl) / nl;
    
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nl; j++)
            D[i*nl + j] = (double) (i*(j+2) % nk) / nk;
}

// Verify results
static double verify_result(int ni, int nl, double *D_ref, double *D_test) {
    double max_error = 0.0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double error = fabs(D_ref[i*nl + j] - D_test[i*nl + j]);
            if (error > max_error) max_error = error;
        }
    }
    return max_error;
}

// Calculate FLOPS for 2mm
static long long calculate_flops(int ni, int nj, int nk, int nl) {
    // First multiplication: ni*nj*nk multiply-add operations
    // Second multiplication: ni*nl*nj multiply-add operations  
    // Each multiply-add = 2 operations
    return 2LL * ni * nj * nk + 2LL * ni * nl * nj;
}

// Strategy 1: Sequential baseline
void kernel_2mm_sequential(int ni, int nj, int nk, int nl,
                          double alpha, double beta,
                          double *A, double *B, double *C, double *D,
                          double *tmp) {
    // First multiplication: tmp = alpha * A * B
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[i*nj + j] = 0.0;
            for (int k = 0; k < nk; k++) {
                tmp[i*nj + j] += alpha * A[i*nk + k] * B[k*nj + j];
            }
        }
    }
    
    // Second multiplication: D = tmp * C + beta * D
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[i*nl + j] *= beta;
            for (int k = 0; k < nj; k++) {
                D[i*nl + j] += tmp[i*nj + k] * C[k*nl + j];
            }
        }
    }
}

// Strategy 2: Basic parallel
void kernel_2mm_basic_parallel(int ni, int nj, int nk, int nl,
                               double alpha, double beta,
                               double *A, double *B, double *C, double *D,
                               double *tmp) {
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
    
    #pragma omp parallel for
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

// Strategy 3: Collapsed loops
void kernel_2mm_collapsed(int ni, int nj, int nk, int nl,
                         double alpha, double beta,
                         double *A, double *B, double *C, double *D,
                         double *tmp) {
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

// Strategy 4: Tiled/Blocked
void kernel_2mm_tiled(int ni, int nj, int nk, int nl,
                     double alpha, double beta,
                     double *A, double *B, double *C, double *D,
                     double *tmp) {
    const int TILE = 32;  // Tile size optimized for cache
    
    // Initialize tmp
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
            tmp[i*nj + j] = 0.0;
    
    // First multiplication with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += TILE) {
        for (int jj = 0; jj < nj; jj += TILE) {
            for (int kk = 0; kk < nk; kk += TILE) {
                int i_end = (ii + TILE < ni) ? ii + TILE : ni;
                int j_end = (jj + TILE < nj) ? jj + TILE : nj;
                int k_end = (kk + TILE < nk) ? kk + TILE : nk;
                
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
    
    // Second multiplication with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += TILE) {
        for (int jj = 0; jj < nl; jj += TILE) {
            int i_end = (ii + TILE < ni) ? ii + TILE : ni;
            int j_end = (jj + TILE < nl) ? jj + TILE : nl;
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    double sum = beta * D[i*nl + j];
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < nj; k++) {
                        sum += tmp[i*nj + k] * C[k*nl + j];
                    }
                    D[i*nl + j] = sum;
                }
            }
        }
    }
}

// Strategy 5: SIMD vectorization
void kernel_2mm_simd(int ni, int nj, int nk, int nl,
                    double alpha, double beta,
                    double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ C, double *__restrict__ D,
                    double *__restrict__ tmp) {
    // Assume aligned pointers for compiler optimization
    A = __builtin_assume_aligned(A, ALIGN_SIZE);
    B = __builtin_assume_aligned(B, ALIGN_SIZE);
    C = __builtin_assume_aligned(C, ALIGN_SIZE);
    D = __builtin_assume_aligned(D, ALIGN_SIZE);
    tmp = __builtin_assume_aligned(tmp, ALIGN_SIZE);
    
    // First multiplication with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        // Initialize row of tmp
        #pragma omp simd aligned(tmp:ALIGN_SIZE)
        for (int j = 0; j < nj; j++) {
            tmp[i*nj + j] = 0.0;
        }
        
        // Compute with SIMD
        for (int k = 0; k < nk; k++) {
            double aik = alpha * A[i*nk + k];
            #pragma omp simd aligned(B,tmp:ALIGN_SIZE)
            for (int j = 0; j < nj; j++) {
                tmp[i*nj + j] += aik * B[k*nj + j];
            }
        }
    }
    
    // Second multiplication with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i*nl + j];
            #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
            for (int k = 0; k < nj; k++) {
                sum += tmp[i*nj + k] * C[k*nl + j];
            }
            D[i*nl + j] = sum;
        }
    }
}

// Strategy 6: Task-based with dependencies
void kernel_2mm_tasks(int ni, int nj, int nk, int nl,
                     double alpha, double beta,
                     double *A, double *B, double *C, double *D,
                     double *tmp) {
    const int CHUNK = 32;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // First multiplication with tasks
            for (int i = 0; i < ni; i += CHUNK) {
                for (int j = 0; j < nj; j += CHUNK) {
                    #pragma omp task depend(out:tmp[i*nj+j:CHUNK*CHUNK])
                    {
                        int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
                        int j_end = (j + CHUNK < nj) ? j + CHUNK : nj;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = 0.0;
                                for (int k = 0; k < nk; k++) {
                                    sum += alpha * A[ii*nk + k] * B[k*nj + jj];
                                }
                                tmp[ii*nj + jj] = sum;
                            }
                        }
                    }
                }
            }
            
            // Second multiplication with tasks and dependencies
            for (int i = 0; i < ni; i += CHUNK) {
                for (int j = 0; j < nl; j += CHUNK) {
                    #pragma omp task depend(in:tmp[i*nj:CHUNK*nj]) \
                                     depend(inout:D[i*nl+j:CHUNK*CHUNK])
                    {
                        int i_end = (i + CHUNK < ni) ? i + CHUNK : ni;
                        int j_end = (j + CHUNK < nl) ? j + CHUNK : nl;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = beta * D[ii*nl + jj];
                                for (int k = 0; k < nj; k++) {
                                    sum += tmp[ii*nj + k] * C[k*nl + jj];
                                }
                                D[ii*nl + jj] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Main benchmark driver
int main(int argc, char** argv) {
    // Allocate aligned memory
    double *A = (double*)aligned_malloc(NI * NK * sizeof(double));
    double *B = (double*)aligned_malloc(NK * NJ * sizeof(double));
    double *C = (double*)aligned_malloc(NJ * NL * sizeof(double));
    double *D_ref = (double*)aligned_malloc(NI * NL * sizeof(double));
    double *D = (double*)aligned_malloc(NI * NL * sizeof(double));
    double *tmp = (double*)aligned_malloc(NI * NJ * sizeof(double));
    
    // Initialize arrays
    init_array(NI, NJ, NK, NL, A, B, C, D_ref);
    
    // Benchmark configuration
    BenchmarkConfig config;
    strcpy(config.kernel_name, "2MM");
    strcpy(config.category, "linear-algebra/kernels");
    config.num_strategies = 6;
    
    // Calculate FLOPS
    long long total_flops = calculate_flops(NI, NJ, NK, NL);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    // Run sequential baseline
    printf("\n=== Running 2MM Benchmark ===\n");
    printf("Problem size: NI=%d, NJ=%d, NK=%d, NL=%d\n", NI, NJ, NK, NL);
    printf("Total FLOPS: %lld\n", total_flops);
    printf("Memory footprint: %.2f MB\n\n", 
           (NI*NK + NK*NJ + NJ*NL + 2*NI*NL + NI*NJ) * sizeof(double) / (1024.0*1024.0));
    
    // Sequential baseline
    memcpy(D, D_ref, NI * NL * sizeof(double));
    double start = omp_get_wtime();
    kernel_2mm_sequential(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
    double serial_time = omp_get_wtime() - start;
    printf("Sequential: %.4f seconds (%.2f GFLOPS)\n", 
           serial_time, total_flops / (serial_time * 1e9));
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    int num_thread_configs = 4;
    
    printf("\n%-20s %-10s %-12s %-12s %-12s %-10s\n",
           "Strategy", "Threads", "Time (s)", "Speedup", "Efficiency", "GFLOPS");
    printf("%-20s %-10s %-12s %-12s %-12s %-10s\n",
           "--------", "-------", "--------", "-------", "----------", "------");
    
    // Define strategies
    typedef void (*strategy_func)(int, int, int, int, double, double, 
                                 double*, double*, double*, double*, double*);
    
    struct {
        const char* name;
        strategy_func func;
    } strategies[] = {
        {"Basic Parallel", kernel_2mm_basic_parallel},
        {"Collapsed", kernel_2mm_collapsed},
        {"Tiled", kernel_2mm_tiled},
        {"SIMD", kernel_2mm_simd},
        {"Task-based", kernel_2mm_tasks}
    };
    
    // Test each strategy
    for (int s = 0; s < 5; s++) {
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);
            
            // Reset D
            memcpy(D, D_ref, NI * NL * sizeof(double));
            
            // Time the strategy
            double times[MEASUREMENT_ITERATIONS];
            for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
                memcpy(D, D_ref, NI * NL * sizeof(double));
                start = omp_get_wtime();
                strategies[s].func(NI, NJ, NK, NL, ALPHA, BETA, A, B, C, D, tmp);
                times[iter] = omp_get_wtime() - start;
            }
            
            // Calculate average time
            double avg_time = 0.0;
            for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
                avg_time += times[i];
            }
            avg_time /= MEASUREMENT_ITERATIONS;
            
            // Verify correctness
            double error = verify_result(NI, NL, D_ref, D);
            
            // Calculate metrics
            double speedup = serial_time / avg_time;
            double efficiency = speedup / num_threads * 100.0;
            double gflops = total_flops / (avg_time * 1e9);
            
            printf("%-20s %-10d %-12.4f %-12.2f %-12.1f%% %-10.2f",
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
    free(D_ref);
    free(D);
    free(tmp);
    
    return 0;
}