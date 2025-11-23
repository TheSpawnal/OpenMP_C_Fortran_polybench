/*
 * Cholesky Decomposition Benchmark - IMPROVED VERSION
 * Enhanced with best practices from project knowledge base
 * Optimizations: better task granularity, reduced synchronization, aligned memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#define TOL 1e-6
#define ALIGN_SIZE 64
#define WARMUP_ITER 2

// Timing function
double wtime() {
    struct timeval time_data;
    gettimeofday(&time_data, NULL);
    return (double)time_data.tv_sec + (double)time_data.tv_usec * 1.0e-6;
}

// Aligned allocation
void* alloc_aligned(size_t size) {
    void *ptr = aligned_alloc(ALIGN_SIZE, size);
    if (!ptr) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Strategy 1: Right-looking with optimal loop ordering
void cholesky_right_looking_opt(int n, double * __restrict__ A, 
                                double * __restrict__ L) {
    memcpy(L, A, n * n * sizeof(double));
    
    for (int k = 0; k < n; k++) {
        // Compute L[k][k]
        double lkk = L[k * n + k];
        if (lkk <= 0.0) {
            fprintf(stderr, "Matrix not positive definite\n");
            return;
        }
        lkk = sqrt(lkk);
        L[k * n + k] = lkk;
        
        // Scale column k (vectorized)
        double inv_lkk = 1.0 / lkk;
        #pragma omp parallel for simd schedule(static)
        for (int i = k + 1; i < n; i++) {
            L[i * n + k] *= inv_lkk;
        }
        
        // Update trailing submatrix with better cache usage
        #pragma omp parallel for schedule(dynamic, 16)
        for (int j = k + 1; j < n; j++) {
            double ljk = L[j * n + k];
            #pragma omp simd
            for (int i = j; i < n; i++) {
                L[i * n + j] -= L[i * n + k] * ljk;
            }
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 2: Blocked algorithm with optimal block size
void cholesky_blocked_opt(int n, double * __restrict__ A, 
                          double * __restrict__ L, int nb) {
    memcpy(L, A, n * n * sizeof(double));
    
    for (int k = 0; k < n; k += nb) {
        int kb = (k + nb < n) ? nb : n - k;
        
        // Factor diagonal block
        for (int kk = 0; kk < kb; kk++) {
            int idx = k + kk;
            
            // Compute L[idx][idx]
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int j = k; j < idx; j++) {
                sum += L[idx * n + j] * L[idx * n + j];
            }
            L[idx * n + idx] = sqrt(L[idx * n + idx] - sum);
            
            // Update column within block
            for (int i = idx + 1; i < k + kb; i++) {
                double s = 0.0;
                #pragma omp simd reduction(+:s)
                for (int j = k; j < idx; j++) {
                    s += L[i * n + j] * L[idx * n + j];
                }
                L[i * n + idx] = (L[i * n + idx] - s) / L[idx * n + idx];
            }
        }
        
        if (k + kb < n) {
            // Solve for panel
            #pragma omp parallel for
            for (int i = k + kb; i < n; i++) {
                for (int j = k; j < k + kb; j++) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int kk = k; kk < j; kk++) {
                        sum += L[i * n + kk] * L[j * n + kk];
                    }
                    L[i * n + j] = (L[i * n + j] - sum) / L[j * n + j];
                }
            }
            
            // Update trailing matrix
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int j = k + kb; j < n; j += nb) {
                for (int i = j; i < n; i += nb) {
                    int jb = (j + nb < n) ? nb : n - j;
                    int ib = (i + nb < n) ? nb : n - i;
                    
                    // Block matrix multiply
                    for (int ii = i; ii < i + ib && ii < n; ii++) {
                        for (int jj = j; jj < j + jb && jj <= ii; jj++) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (int kk = k; kk < k + kb; kk++) {
                                sum += L[ii * n + kk] * L[jj * n + kk];
                            }
                            L[ii * n + jj] -= sum;
                        }
                    }
                }
            }
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 3: Task-based with improved dependencies
void cholesky_tasks_opt(int n, double * __restrict__ A,
                        double * __restrict__ L, int bs) {
    memcpy(L, A, n * n * sizeof(double));
    
    int nt = (n + bs - 1) / bs;  // Number of tile rows/cols
    
    #pragma omp parallel
    #pragma omp single
    {
        for (int k = 0; k < nt; k++) {
            int k_start = k * bs;
            int k_size = (k_start + bs < n) ? bs : n - k_start;
            
            // Cholesky factorization of diagonal tile
            #pragma omp task depend(inout:L[k_start*n+k_start:k_size*n])
            {
                for (int j = k_start; j < k_start + k_size; j++) {
                    // Compute L[j][j]
                    double sum = 0.0;
                    for (int i = k_start; i < j; i++) {
                        sum += L[j * n + i] * L[j * n + i];
                    }
                    L[j * n + j] = sqrt(L[j * n + j] - sum);
                    
                    // Update column j
                    for (int i = j + 1; i < k_start + k_size; i++) {
                        double s = 0.0;
                        for (int kk = k_start; kk < j; kk++) {
                            s += L[i * n + kk] * L[j * n + kk];
                        }
                        L[i * n + j] = (L[i * n + j] - s) / L[j * n + j];
                    }
                }
            }
            
            // Triangular solve for panel
            for (int i = k + 1; i < nt; i++) {
                int i_start = i * bs;
                int i_size = (i_start + bs < n) ? bs : n - i_start;
                
                #pragma omp task depend(in:L[k_start*n+k_start:k_size*n]) \
                                 depend(inout:L[i_start*n+k_start:i_size*k_size])
                {
                    for (int ii = i_start; ii < i_start + i_size; ii++) {
                        for (int jj = k_start; jj < k_start + k_size; jj++) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (int kk = k_start; kk < jj; kk++) {
                                sum += L[ii * n + kk] * L[jj * n + kk];
                            }
                            L[ii * n + jj] = (L[ii * n + jj] - sum) / L[jj * n + jj];
                        }
                    }
                }
            }
            
            // Symmetric rank-k update
            for (int j = k + 1; j < nt; j++) {
                int j_start = j * bs;
                int j_size = (j_start + bs < n) ? bs : n - j_start;
                
                for (int i = j; i < nt; i++) {
                    int i_start = i * bs;
                    int i_size = (i_start + bs < n) ? bs : n - i_start;
                    
                    #pragma omp task depend(in:L[i_start*n+k_start:i_size*k_size], \
                                              L[j_start*n+k_start:j_size*k_size]) \
                                     depend(inout:L[i_start*n+j_start:i_size*j_size])
                    {
                        for (int ii = i_start; ii < i_start + i_size; ii++) {
                            int jj_end = (i == j) ? ii + 1 : j_start + j_size;
                            for (int jj = j_start; jj < jj_end && jj < n; jj++) {
                                double sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (int kk = k_start; kk < k_start + k_size; kk++) {
                                    sum += L[ii * n + kk] * L[jj * n + kk];
                                }
                                L[ii * n + jj] -= sum;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 4: Recursive with improved base case
void cholesky_recursive_impl(int n, double * __restrict__ A, int lda, int depth) {
    const int BASE_SIZE = 64;  // Optimized base case size
    
    if (n <= BASE_SIZE) {
        // Optimized base case with SIMD
        for (int j = 0; j < n; j++) {
            // Compute diagonal element
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < j; k++) {
                sum += A[j * lda + k] * A[j * lda + k];
            }
            A[j * lda + j] = sqrt(A[j * lda + j] - sum);
            
            // Update column
            double inv_ajj = 1.0 / A[j * lda + j];
            #pragma omp simd
            for (int i = j + 1; i < n; i++) {
                double s = 0.0;
                for (int k = 0; k < j; k++) {
                    s += A[i * lda + k] * A[j * lda + k];
                }
                A[i * lda + j] = (A[i * lda + j] - s) * inv_ajj;
            }
        }
        return;
    }
    
    int n1 = n / 2;
    int n2 = n - n1;
    
    // Cholesky of A11
    cholesky_recursive_impl(n1, A, lda, depth + 1);
    
    // Solve for A21 (parallel if at top levels)
    #pragma omp parallel for if(depth < 2)
    for (int i = n1; i < n; i++) {
        for (int j = 0; j < n1; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < j; k++) {
                sum += A[i * lda + k] * A[j * lda + k];
            }
            A[i * lda + j] = (A[i * lda + j] - sum) / A[j * lda + j];
        }
    }
    
    // Update A22 (parallel if at top levels)
    #pragma omp parallel for collapse(2) if(depth < 2)
    for (int i = n1; i < n; i++) {
        for (int j = n1; j <= i; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n1; k++) {
                sum += A[i * lda + k] * A[j * lda + k];
            }
            A[i * lda + j] -= sum;
        }
    }
    
    // Cholesky of A22
    cholesky_recursive_impl(n2, &A[n1 * lda + n1], lda, depth + 1);
}

void cholesky_recursive_opt(int n, double * __restrict__ A, 
                            double * __restrict__ L) {
    memcpy(L, A, n * n * sizeof(double));
    cholesky_recursive_impl(n, L, n, 0);
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 5: Left-looking with prefetching
void cholesky_left_looking_opt(int n, double * __restrict__ A,
                               double * __restrict__ L) {
    memcpy(L, A, n * n * sizeof(double));
    
    for (int j = 0; j < n; j++) {
        // Prefetch next column
        if (j + 1 < n) {
            for (int i = j + 1; i < n && i < j + 16; i++) {
                __builtin_prefetch(&L[i * n + j + 1], 0, 3);
            }
        }
        
        // Update column j using previous columns
        #pragma omp parallel for schedule(static) if(n - j > 128)
        for (int i = j; i < n; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] -= sum;
        }
        
        // Compute diagonal element
        if (L[j * n + j] <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at %d\n", j);
            return;
        }
        L[j * n + j] = sqrt(L[j * n + j]);
        
        // Scale column j
        double inv_ljj = 1.0 / L[j * n + j];
        #pragma omp parallel for simd schedule(static)
        for (int i = j + 1; i < n; i++) {
            L[i * n + j] *= inv_ljj;
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Generate positive definite matrix
void generate_posdef_matrix(int n, double *A) {
    unsigned int seed = 42;
    
    // Create random symmetric matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            seed = seed * 1103515245 + 12345;
            double val = (seed / 65536) % 100;
            A[i * n + j] = val;
            A[j * n + i] = val;
        }
        // Make diagonally dominant
        A[i * n + i] += n * 100;
    }
}

// Verify Cholesky decomposition
double verify_cholesky(int n, double *A, double *L) {
    double max_error = 0.0;
    
    #pragma omp parallel for reduction(max:max_error)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            int k_end = (j < n) ? j + 1 : n;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < k_end; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            double error = fabs(A[i * n + j] - sum);
            max_error = fmax(max_error, error);
        }
    }
    return max_error;
}

int main(int argc, char **argv) {
    int n = 1500;
    int nthreads = omp_get_max_threads();
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) {
        nthreads = atoi(argv[2]);
        omp_set_num_threads(nthreads);
    }
    
    printf("===== Improved Cholesky Decomposition Benchmark =====\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Threads: %d\n", nthreads);
    printf("Memory: %.2f MB\n\n", (n * n * sizeof(double)) / (1024.0 * 1024.0));
    
    // Allocate aligned matrices
    double *A = (double *)alloc_aligned(n * n * sizeof(double));
    double *L = (double *)alloc_aligned(n * n * sizeof(double));
    
    // Generate positive definite matrix
    generate_posdef_matrix(n, A);
    
    double start_time, end_time, error;
    double best_time = 1e9;
    const char *best_method = "";
    
    // Warmup
    printf("Warming up...\n");
    for (int w = 0; w < WARMUP_ITER; w++) {
        cholesky_right_looking_opt(n, A, L);
    }
    
    printf("\nPerformance Results:\n");
    printf("%-25s %10s %12s %10s\n", "Method", "Time (s)", "GFLOP/s", "Error");
    printf("%-25s %10s %12s %10s\n", "------", "--------", "-------", "-----");
    
    double flops = (n * n * n) / 3.0;  // Approximate FLOPs for Cholesky
    
    // Test all strategies
    start_time = wtime();
    cholesky_right_looking_opt(n, A, L);
    end_time = wtime();
    double time1 = end_time - start_time;
    error = verify_cholesky(n, A, L);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Right-looking Optimized", time1, (flops/1e9)/time1, error);
    if (time1 < best_time) { best_time = time1; best_method = "Right-looking"; }
    
    start_time = wtime();
    cholesky_blocked_opt(n, A, L, 128);
    end_time = wtime();
    double time2 = end_time - start_time;
    error = verify_cholesky(n, A, L);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Blocked (128)", time2, (flops/1e9)/time2, error);
    if (time2 < best_time) { best_time = time2; best_method = "Blocked"; }
    
    start_time = wtime();
    cholesky_tasks_opt(n, A, L, 128);
    end_time = wtime();
    double time3 = end_time - start_time;
    error = verify_cholesky(n, A, L);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Tasks Optimized (128)", time3, (flops/1e9)/time3, error);
    if (time3 < best_time) { best_time = time3; best_method = "Tasks"; }
    
    start_time = wtime();
    cholesky_recursive_opt(n, A, L);
    end_time = wtime();
    double time4 = end_time - start_time;
    error = verify_cholesky(n, A, L);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Recursive Optimized", time4, (flops/1e9)/time4, error);
    if (time4 < best_time) { best_time = time4; best_method = "Recursive"; }
    
    start_time = wtime();
    cholesky_left_looking_opt(n, A, L);
    end_time = wtime();
    double time5 = end_time - start_time;
    error = verify_cholesky(n, A, L);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Left-looking Optimized", time5, (flops/1e9)/time5, error);
    if (time5 < best_time) { best_time = time5; best_method = "Left-looking"; }
    
    printf("\nBest method: %s (%.4f seconds)\n", best_method, best_time);
    printf("Speedup vs baseline: %.2fx\n", time1 / best_time);
    
    free(A);
    free(L);
    
    return 0;
}
