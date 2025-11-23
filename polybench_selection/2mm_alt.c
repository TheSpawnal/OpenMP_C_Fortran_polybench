/*
 * 2mm Benchmark - IMPROVED VERSION
 * Double Matrix Multiplication: D = alpha*A*B*C + beta*D
 * Enhanced with best practices from project knowledge base
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define ALPHA 1.5
#define BETA 1.2
#define TOL 0.001
#define ALIGN_SIZE 64
#define WARMUP_ITER 2

// Timing function
double wtime() {
    struct timeval time_data;
    gettimeofday(&time_data, NULL);
    return (double)time_data.tv_sec + (double)time_data.tv_usec * 1.0e-6;
}

// Aligned allocation with error checking
void* alloc_aligned(size_t size) {
    void *ptr = aligned_alloc(ALIGN_SIZE, size);
    if (!ptr) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Strategy 1: Optimized with loop reordering (ikj pattern from project)
void mm2_ikj_optimized(int ni, int nj, int nk, int nl,
                      double * __restrict__ A, double * __restrict__ B, 
                      double * __restrict__ C, double * __restrict__ D,
                      double * __restrict__ tmp, double alpha, double beta) {
    
    // First multiplication: tmp = alpha * A * B (using ikj order)
    memset(tmp, 0, ni * nj * sizeof(double));
    
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int k = 0; k < nk; k++) {
            double aik = alpha * A[i * nk + k];
            #pragma omp simd
            for (int j = 0; j < nj; j++) {
                tmp[i * nj + j] += aik * B[k * nj + j];
            }
        }
    }
    
    // Second multiplication: D = tmp * C + beta * D (using ikj order)
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[i * nl + j] *= beta;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int k = 0; k < nj; k++) {
            double tik = tmp[i * nj + k];
            #pragma omp simd
            for (int j = 0; j < nl; j++) {
                D[i * nl + j] += tik * C[k * nl + j];
            }
        }
    }
}

// Strategy 2: Tiled with optimal cache blocking
void mm2_tiled_cache(int ni, int nj, int nk, int nl,
                     double * __restrict__ A, double * __restrict__ B,
                     double * __restrict__ C, double * __restrict__ D,
                     double * __restrict__ tmp, double alpha, double beta,
                     int tile_size) {
    
    // Initialize tmp
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[i * nj + j] = 0.0;
        }
    }
    
    // First multiplication with 3-level tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += tile_size) {
        for (int jj = 0; jj < nj; jj += tile_size) {
            int i_end = (ii + tile_size < ni) ? ii + tile_size : ni;
            int j_end = (jj + tile_size < nj) ? jj + tile_size : nj;
            
            for (int kk = 0; kk < nk; kk += tile_size) {
                int k_end = (kk + tile_size < nk) ? kk + tile_size : nk;
                
                // Micro-kernel with register blocking
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double aik = alpha * A[i * nk + k];
                        #pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            tmp[i * nj + j] += aik * B[k * nj + j];
                        }
                    }
                }
            }
        }
    }
    
    // Apply beta scaling
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            D[i * nl + j] *= beta;
        }
    }
    
    // Second multiplication with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += tile_size) {
        for (int jj = 0; jj < nl; jj += tile_size) {
            int i_end = (ii + tile_size < ni) ? ii + tile_size : ni;
            int j_end = (jj + tile_size < nl) ? jj + tile_size : nl;
            
            for (int kk = 0; kk < nj; kk += tile_size) {
                int k_end = (kk + tile_size < nj) ? kk + tile_size : nj;
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double tik = tmp[i * nj + k];
                        #pragma omp simd
                        for (int j = jj; j < j_end; j++) {
                            D[i * nl + j] += tik * C[k * nl + j];
                        }
                    }
                }
            }
        }
    }
}

// Strategy 3: SIMD-optimized with aligned access
void mm2_simd_aligned(int ni, int nj, int nk, int nl,
                      double * __restrict__ A, double * __restrict__ B,
                      double * __restrict__ C, double * __restrict__ D,
                      double * __restrict__ tmp, double alpha, double beta) {
    
    // Ensure aligned pointers
    __assume_aligned(A, ALIGN_SIZE);
    __assume_aligned(B, ALIGN_SIZE);
    __assume_aligned(C, ALIGN_SIZE);
    __assume_aligned(D, ALIGN_SIZE);
    __assume_aligned(tmp, ALIGN_SIZE);
    
    // First multiplication with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        // Initialize row of tmp
        #pragma omp simd aligned(tmp:ALIGN_SIZE)
        for (int j = 0; j < nj; j++) {
            tmp[i * nj + j] = 0.0;
        }
        
        // Compute dot products
        for (int k = 0; k < nk; k++) {
            double aik = alpha * A[i * nk + k];
            #pragma omp simd aligned(B,tmp:ALIGN_SIZE)
            for (int j = 0; j < nj; j++) {
                tmp[i * nj + j] += aik * B[k * nj + j];
            }
        }
    }
    
    // Second multiplication with SIMD
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i * nl + j];
            #pragma omp simd reduction(+:sum) aligned(tmp,C:ALIGN_SIZE)
            for (int k = 0; k < nj; k++) {
                sum += tmp[i * nj + k] * C[k * nl + j];
            }
            D[i * nl + j] = sum;
        }
    }
}

// Strategy 4: Task-based with dependency tracking
void mm2_tasks_deps(int ni, int nj, int nk, int nl,
                   double * __restrict__ A, double * __restrict__ B,
                   double * __restrict__ C, double * __restrict__ D,
                   double * __restrict__ tmp, double alpha, double beta,
                   int chunk) {
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Initialize tmp with tasks
            for (int i = 0; i < ni; i += chunk) {
                #pragma omp task depend(out:tmp[i*nj:(chunk*nj)])
                {
                    int i_end = (i + chunk < ni) ? i + chunk : ni;
                    for (int ii = i; ii < i_end; ii++) {
                        for (int j = 0; j < nj; j++) {
                            tmp[ii * nj + j] = 0.0;
                        }
                    }
                }
            }
            
            // First multiplication with tasks and dependencies
            for (int i = 0; i < ni; i += chunk) {
                for (int j = 0; j < nj; j += chunk) {
                    for (int k = 0; k < nk; k += chunk) {
                        #pragma omp task depend(in:A[i*nk:chunk*nk], B[k*nj:chunk*nj]) \
                                         depend(inout:tmp[i*nj:chunk*nj])
                        {
                            int i_end = (i + chunk < ni) ? i + chunk : ni;
                            int j_end = (j + chunk < nj) ? j + chunk : nj;
                            int k_end = (k + chunk < nk) ? k + chunk : nk;
                            
                            for (int ii = i; ii < i_end; ii++) {
                                for (int kk = k; kk < k_end; kk++) {
                                    double aik = alpha * A[ii * nk + kk];
                                    for (int jj = j; jj < j_end; jj++) {
                                        tmp[ii * nj + jj] += aik * B[kk * nj + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Scale D with beta
            for (int i = 0; i < ni; i += chunk) {
                #pragma omp task depend(inout:D[i*nl:chunk*nl])
                {
                    int i_end = (i + chunk < ni) ? i + chunk : ni;
                    for (int ii = i; ii < i_end; ii++) {
                        for (int j = 0; j < nl; j++) {
                            D[ii * nl + j] *= beta;
                        }
                    }
                }
            }
            
            // Second multiplication with tasks
            for (int i = 0; i < ni; i += chunk) {
                for (int j = 0; j < nl; j += chunk) {
                    for (int k = 0; k < nj; k += chunk) {
                        #pragma omp task depend(in:tmp[i*nj:chunk*nj], C[k*nl:chunk*nl]) \
                                         depend(inout:D[i*nl:chunk*nl])
                        {
                            int i_end = (i + chunk < ni) ? i + chunk : ni;
                            int j_end = (j + chunk < nl) ? j + chunk : nl;
                            int k_end = (k + chunk < nj) ? k + chunk : nj;
                            
                            for (int ii = i; ii < i_end; ii++) {
                                for (int kk = k; kk < k_end; kk++) {
                                    double tik = tmp[ii * nj + kk];
                                    for (int jj = j; jj < j_end; jj++) {
                                        D[ii * nl + jj] += tik * C[kk * nl + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Strategy 5: Hierarchical parallelism with nested regions
void mm2_hierarchical(int ni, int nj, int nk, int nl,
                     double * __restrict__ A, double * __restrict__ B,
                     double * __restrict__ C, double * __restrict__ D,
                     double * __restrict__ tmp, double alpha, double beta) {
    
    int outer_threads = omp_get_max_threads();
    int inner_threads = 1;
    
    // Adjust for nested parallelism if supported
    if (omp_get_nested()) {
        outer_threads = (int)sqrt(omp_get_max_threads());
        inner_threads = omp_get_max_threads() / outer_threads;
    }
    
    // First multiplication with hierarchical parallelism
    #pragma omp parallel for num_threads(outer_threads)
    for (int i = 0; i < ni; i++) {
        // Initialize row
        for (int j = 0; j < nj; j++) {
            tmp[i * nj + j] = 0.0;
        }
        
        // Inner parallel region for dot products
        #pragma omp parallel for num_threads(inner_threads) if(inner_threads > 1)
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nk; k++) {
                sum += alpha * A[i * nk + k] * B[k * nj + j];
            }
            tmp[i * nj + j] = sum;
        }
    }
    
    // Second multiplication
    #pragma omp parallel for num_threads(outer_threads)
    for (int i = 0; i < ni; i++) {
        #pragma omp parallel for num_threads(inner_threads) if(inner_threads > 1)
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i * nl + j];
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nj; k++) {
                sum += tmp[i * nj + k] * C[k * nl + j];
            }
            D[i * nl + j] = sum;
        }
    }
}

// Initialize matrices
void init_matrices(int ni, int nj, int nk, int nl,
                  double *A, double *B, double *C, double *D) {
    unsigned int seed = 42;
    
    #pragma omp parallel for firstprivate(seed)
    for (int i = 0; i < ni * nk; i++) {
        seed = seed * 1103515245 + 12345 + i;
        A[i] = (seed / 65536) % 100;
    }
    
    #pragma omp parallel for firstprivate(seed)
    for (int i = 0; i < nk * nj; i++) {
        seed = seed * 1103515245 + 12345 + i + 1000;
        B[i] = (seed / 65536) % 100;
    }
    
    #pragma omp parallel for firstprivate(seed)
    for (int i = 0; i < nj * nl; i++) {
        seed = seed * 1103515245 + 12345 + i + 2000;
        C[i] = (seed / 65536) % 100;
    }
    
    #pragma omp parallel for firstprivate(seed)
    for (int i = 0; i < ni * nl; i++) {
        seed = seed * 1103515245 + 12345 + i + 3000;
        D[i] = (seed / 65536) % 100;
    }
}

// Verify result
double verify_result(int ni, int nl, double *D1, double *D2) {
    double max_diff = 0.0;
    
    #pragma omp parallel for reduction(max:max_diff)
    for (int i = 0; i < ni * nl; i++) {
        double diff = fabs(D1[i] - D2[i]);
        max_diff = fmax(max_diff, diff);
    }
    
    return max_diff;
}

int main(int argc, char **argv) {
    int ni = 800, nj = 900, nk = 1000, nl = 1100;
    int nthreads = omp_get_max_threads();
    
    if (argc > 1) ni = atoi(argv[1]);
    if (argc > 2) nj = atoi(argv[2]);
    if (argc > 3) nk = atoi(argv[3]);
    if (argc > 4) nl = atoi(argv[4]);
    if (argc > 5) {
        nthreads = atoi(argv[5]);
        omp_set_num_threads(nthreads);
    }
    
    printf("===== Improved 2mm Benchmark =====\n");
    printf("Matrix dimensions: A(%d,%d) B(%d,%d) C(%d,%d) D(%d,%d)\n",
           ni, nk, nk, nj, nj, nl, ni, nl);
    printf("Threads: %d\n", nthreads);
    
    double memory_mb = ((ni*nk + nk*nj + nj*nl + ni*nl + ni*nj) * sizeof(double)) / (1024.0 * 1024.0);
    printf("Total memory: %.2f MB\n", memory_mb);
    
    double flops = 2.0 * ni * nj * nk + 2.0 * ni * nl * nj + ni * nl;
    printf("Total FLOPs: %.2f GFLOPs\n\n", flops / 1e9);
    
    // Allocate aligned matrices
    double *A = (double *)alloc_aligned(ni * nk * sizeof(double));
    double *B = (double *)alloc_aligned(nk * nj * sizeof(double));
    double *C = (double *)alloc_aligned(nj * nl * sizeof(double));
    double *D = (double *)alloc_aligned(ni * nl * sizeof(double));
    double *D_ref = (double *)alloc_aligned(ni * nl * sizeof(double));
    double *tmp = (double *)alloc_aligned(ni * nj * sizeof(double));
    
    init_matrices(ni, nj, nk, nl, A, B, C, D);
    memcpy(D_ref, D, ni * nl * sizeof(double));
    
    double start_time, end_time;
    double best_time = 1e9;
    const char *best_method = "";
    
    // Warmup
    printf("Warming up...\n");
    for (int w = 0; w < WARMUP_ITER; w++) {
        mm2_ikj_optimized(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    }
    
    printf("\nPerformance Results:\n");
    printf("%-25s %10s %12s %10s\n", "Method", "Time (s)", "GFLOP/s", "Error");
    printf("%-25s %10s %12s %10s\n", "------", "--------", "-------", "-----");
    
    // Test Strategy 1: IKJ optimized
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = wtime();
    mm2_ikj_optimized(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    end_time = wtime();
    double time1 = end_time - start_time;
    double gflops1 = (flops / 1e9) / time1;
    printf("%-25s %10.4f %12.2f %10.2e\n", "IKJ Optimized", time1, gflops1, 0.0);
    if (time1 < best_time) { best_time = time1; best_method = "IKJ Optimized"; }
    double *D_verify = (double *)alloc_aligned(ni * nl * sizeof(double));
    memcpy(D_verify, D, ni * nl * sizeof(double));
    
    // Test Strategy 2: Tiled cache-optimized
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = wtime();
    mm2_tiled_cache(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA, 64);
    end_time = wtime();
    double time2 = end_time - start_time;
    double gflops2 = (flops / 1e9) / time2;
    double error2 = verify_result(ni, nl, D, D_verify);
    printf("%-25s %10.4f %12.2f %10.2e\n", "Tiled Cache (64)", time2, gflops2, error2);
    if (time2 < best_time) { best_time = time2; best_method = "Tiled Cache"; }
    
    // Test Strategy 3: SIMD aligned
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = wtime();
    mm2_simd_aligned(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    end_time = wtime();
    double time3 = end_time - start_time;
    double gflops3 = (flops / 1e9) / time3;
    double error3 = verify_result(ni, nl, D, D_verify);
    printf("%-25s %10.4f %12.2f %10.2e\n", "SIMD Aligned", time3, gflops3, error3);
    if (time3 < best_time) { best_time = time3; best_method = "SIMD Aligned"; }
    
    // Test Strategy 4: Tasks with dependencies
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = wtime();
    mm2_tasks_deps(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA, 100);
    end_time = wtime();
    double time4 = end_time - start_time;
    double gflops4 = (flops / 1e9) / time4;
    double error4 = verify_result(ni, nl, D, D_verify);
    printf("%-25s %10.4f %12.2f %10.2e\n", "Tasks with Deps", time4, gflops4, error4);
    if (time4 < best_time) { best_time = time4; best_method = "Tasks with Deps"; }
    
    // Test Strategy 5: Hierarchical
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = wtime();
    mm2_hierarchical(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    end_time = wtime();
    double time5 = end_time - start_time;
    double gflops5 = (flops / 1e9) / time5;
    double error5 = verify_result(ni, nl, D, D_verify);
    printf("%-25s %10.4f %12.2f %10.2e\n", "Hierarchical", time5, gflops5, error5);
    if (time5 < best_time) { best_time = time5; best_method = "Hierarchical"; }
    
    printf("\nBest performing method: %s\n", best_method);
    printf("Best time: %.4f seconds (%.2f GFLOP/s)\n", best_time, (flops / 1e9) / best_time);
    printf("Speedup vs baseline: %.2fx\n", time1 / best_time);
    
    free(A); free(B); free(C); free(D);
    free(D_ref); free(D_verify); free(tmp);
    
    return 0;
}
