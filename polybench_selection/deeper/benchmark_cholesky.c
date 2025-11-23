/**
 * cholesky.c: Cholesky Decomposition Benchmark with Multiple Strategies
 * A = L * L^T decomposition for symmetric positive-definite matrices
 * 
 * Strategies implemented:
 * 1. Sequential (baseline)
 * 2. Column-wise parallel
 * 3. Right-looking blocked
 * 4. Left-looking with prefetching
 * 5. Task-based with dependencies
 * 6. Recursive divide-and-conquer
 * 7. Hybrid (coarse+fine grained)
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
#define N 40
#elif defined(SMALL)
#define N 120
#elif defined(MEDIUM)
#define N 400
#elif defined(LARGE)
#define N 2000
#else // Default STANDARD
#define N 500
#endif

#define ALIGN_SIZE 64
#define BLOCK_SIZE 64

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

// Initialize symmetric positive-definite matrix
static void init_array(int n, double *A) {
    // Generate a symmetric positive-definite matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[i*n + j] = (double) (-j % n) / n + 1;
            A[j*n + i] = A[i*n + j];
        }
        A[i*n + i] += n;  // Ensure diagonal dominance
    }
}

// Verify result by checking L * L^T = A
static double verify_cholesky(int n, double *A_orig, double *L) {
    double max_error = 0.0;
    
    // Compute L * L^T
    double *result = (double*)aligned_malloc(n * n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += L[i*n + k] * L[j*n + k];
            }
            result[i*n + j] = sum;
            if (i != j) result[j*n + i] = sum;
        }
    }
    
    // Compare with original matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double error = fabs(A_orig[i*n + j] - result[i*n + j]);
            if (error > max_error) max_error = error;
        }
    }
    
    free(result);
    return max_error;
}

// Calculate FLOPS for Cholesky
static long long calculate_flops(int n) {
    // Cholesky: ~n^3/3 operations
    return (long long)n * n * n / 3;
}

// Strategy 1: Sequential baseline
void kernel_cholesky_sequential(int n, double *A) {
    for (int i = 0; i < n; i++) {
        // Diagonal element
        for (int j = 0; j < i; j++) {
            A[i*n + i] -= A[i*n + j] * A[i*n + j];
        }
        A[i*n + i] = sqrt(A[i*n + i]);
        
        // Column below diagonal
        for (int j = i + 1; j < n; j++) {
            for (int k = 0; k < i; k++) {
                A[j*n + i] -= A[j*n + k] * A[i*n + k];
            }
            A[j*n + i] /= A[i*n + i];
        }
    }
    
    // Zero upper triangle
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i*n + j] = 0.0;
        }
    }
}

// Strategy 2: Column-wise parallel
void kernel_cholesky_column_parallel(int n, double *A) {
    for (int k = 0; k < n; k++) {
        // Diagonal element (sequential)
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += A[k*n + j] * A[k*n + j];
        }
        A[k*n + k] = sqrt(A[k*n + k] - sum);
        
        // Update column in parallel
        double inv_akk = 1.0 / A[k*n + k];
        
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int j = 0; j < k; j++) {
                s += A[i*n + j] * A[k*n + j];
            }
            A[i*n + k] = (A[i*n + k] - s) * inv_akk;
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i*n + j] = 0.0;
        }
    }
}

// Strategy 3: Right-looking blocked
void kernel_cholesky_right_looking_blocked(int n, double *A) {
    const int bs = BLOCK_SIZE;
    
    for (int k = 0; k < n; k += bs) {
        int bk = (k + bs < n) ? bs : n - k;
        
        // Factorize diagonal block
        for (int i = k; i < k + bk; i++) {
            // Diagonal element
            double sum = 0.0;
            for (int j = k; j < i; j++) {
                sum += A[i*n + j] * A[i*n + j];
            }
            A[i*n + i] = sqrt(A[i*n + i] - sum);
            
            // Update within block
            for (int j = i + 1; j < k + bk; j++) {
                double s = 0.0;
                for (int l = k; l < i; l++) {
                    s += A[j*n + l] * A[i*n + l];
                }
                A[j*n + i] = (A[j*n + i] - s) / A[i*n + i];
            }
        }
        
        // Update trailing matrix
        if (k + bk < n) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = k + bk; i < n; i++) {
                for (int j = k; j < k + bk; j++) {
                    double s = 0.0;
                    #pragma omp simd reduction(+:s)
                    for (int l = 0; l < j; l++) {
                        s += A[i*n + l] * A[j*n + l];
                    }
                    A[i*n + j] = (A[i*n + j] - s) / A[j*n + j];
                }
                
                // Update diagonal block of row i
                for (int j = k + bk; j <= i; j++) {
                    double s = 0.0;
                    #pragma omp simd reduction(+:s)
                    for (int l = k; l < k + bk; l++) {
                        s += A[i*n + l] * A[j*n + l];
                    }
                    A[i*n + j] -= s;
                }
            }
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i*n + j] = 0.0;
        }
    }
}

// Strategy 4: Left-looking with prefetching
void kernel_cholesky_left_looking(int n, double *A) {
    for (int j = 0; j < n; j++) {
        // Prefetch next column
        if (j + 1 < n) {
            for (int i = j + 1; i < n && i < j + 16; i++) {
                __builtin_prefetch(&A[i*n + j + 1], 0, 3);
            }
        }
        
        // Update column j using previous columns
        #pragma omp parallel for schedule(static)
        for (int i = j; i < n; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < j; k++) {
                sum += A[i*n + k] * A[j*n + k];
            }
            A[i*n + j] -= sum;
        }
        
        // Compute diagonal element
        A[j*n + j] = sqrt(A[j*n + j]);
        
        // Scale column
        double inv_ajj = 1.0 / A[j*n + j];
        #pragma omp parallel for simd
        for (int i = j + 1; i < n; i++) {
            A[i*n + j] *= inv_ajj;
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            A[i*n + j] = 0.0;
        }
    }
}

// Strategy 5: Task-based with dependencies
void kernel_cholesky_tasks(int n, double *A) {
    const int bs = BLOCK_SIZE;
    
    #pragma omp parallel
    #pragma omp single
    {
        for (int k = 0; k < n; k += bs) {
            int k_size = (k + bs < n) ? bs : n - k;
            
            // Diagonal block factorization
            #pragma omp task depend(inout:A[k*n+k:k_size*k_size])
            {
                for (int i = k; i < k + k_size; i++) {
                    // Diagonal element
                    double sum = 0.0;
                    for (int j = k; j < i; j++) {
                        sum += A[i*n + j] * A[i*n + j];
                    }
                    A[i*n + i] = sqrt(A[i*n + i] - sum);
                    
                    // Column update within block
                    for (int j = i + 1; j < k + k_size; j++) {
                        double s = 0.0;
                        for (int l = k; l < i; l++) {
                            s += A[j*n + l] * A[i*n + l];
                        }
                        A[j*n + i] = (A[j*n + i] - s) / A[i*n + i];
                    }
                }
            }
            
            // Panel factorization
            for (int i = k + bs; i < n; i += bs) {
                int i_size = (i + bs < n) ? bs : n - i;
                
                #pragma omp task depend(in:A[k*n+k:k_size*k_size]) \
                                 depend(inout:A[i*n+k:i_size*k_size])
                {
                    for (int ii = i; ii < i + i_size; ii++) {
                        for (int jj = k; jj < k + k_size; jj++) {
                            double s = 0.0;
                            for (int kk = 0; kk < jj; kk++) {
                                s += A[ii*n + kk] * A[jj*n + kk];
                            }
                            A[ii*n + jj] = (A[ii*n + jj] - s) / A[jj*n + jj];
                        }
                    }
                }
            }
            
            // Trailing matrix update
            for (int i = k + bs; i < n; i += bs) {
                for (int j = k + bs; j <= i; j += bs) {
                    int i_size = (i + bs < n) ? bs : n - i;
                    int j_size = (j + bs < n) ? bs : n - j;
                    
                    #pragma omp task depend(in:A[i*n+k:i_size*k_size], \
                                              A[j*n+k:j_size*k_size]) \
                                     depend(inout:A[i*n+j:i_size*j_size])
                    {
                        for (int ii = i; ii < i + i_size; ii++) {
                            int jj_end = (i == j) ? ii + 1 : j + j_size;
                            for (int jj = j; jj < jj_end && jj < n; jj++) {
                                double sum = 0.0;
                                for (int kk = k; kk < k + k_size; kk++) {
                                    sum += A[ii*n + kk] * A[jj*n + kk];
                                }
                                A[ii*n + jj] -= sum;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i*n + j] = 0.0;
        }
    }
}

// Strategy 6: Recursive divide-and-conquer
void cholesky_recursive_impl(int n, double *A, int lda, int depth) {
    const int BASE_SIZE = 64;
    
    if (n <= BASE_SIZE) {
        // Base case: sequential Cholesky
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += A[i*lda + k] * A[i*lda + k];
            }
            A[i*lda + i] = sqrt(A[i*lda + i] - sum);
            
            double inv_aii = 1.0 / A[i*lda + i];
            for (int j = i + 1; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < i; k++) {
                    s += A[j*lda + k] * A[i*lda + k];
                }
                A[j*lda + i] = (A[j*lda + i] - s) * inv_aii;
            }
        }
        return;
    }
    
    int n1 = n / 2;
    int n2 = n - n1;
    
    // Cholesky of A11
    cholesky_recursive_impl(n1, A, lda, depth + 1);
    
    // Solve for A21 (parallel at top levels)
    #pragma omp parallel for if(depth < 2)
    for (int i = n1; i < n; i++) {
        for (int j = 0; j < n1; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += A[i*lda + k] * A[j*lda + k];
            }
            A[i*lda + j] = (A[i*lda + j] - sum) / A[j*lda + j];
        }
    }
    
    // Update A22 (parallel at top levels)
    #pragma omp parallel for collapse(2) if(depth < 2)
    for (int i = n1; i < n; i++) {
        for (int j = n1; j <= i; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n1; k++) {
                sum += A[i*lda + k] * A[j*lda + k];
            }
            A[i*lda + j] -= sum;
        }
    }
    
    // Cholesky of A22
    cholesky_recursive_impl(n2, &A[n1*lda + n1], lda, depth + 1);
}

void kernel_cholesky_recursive(int n, double *A) {
    cholesky_recursive_impl(n, A, n, 0);
    
    // Zero upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[i*n + j] = 0.0;
        }
    }
}

// Main benchmark driver
int main(int argc, char** argv) {
    // Allocate aligned memory
    double *A_orig = (double*)aligned_malloc(N * N * sizeof(double));
    double *A = (double*)aligned_malloc(N * N * sizeof(double));
    
    // Initialize matrix
    init_array(N, A_orig);
    
    // Calculate FLOPS
    long long total_flops = calculate_flops(N);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    printf("\n=== Running Cholesky Decomposition Benchmark ===\n");
    printf("Matrix dimension: %d x %d\n", N, N);
    printf("Total FLOPS: %lld\n", total_flops);
    printf("Memory footprint: %.2f MB\n\n", 
           N * N * sizeof(double) / (1024.0*1024.0));
    
    // Sequential baseline
    memcpy(A, A_orig, N * N * sizeof(double));
    double start = omp_get_wtime();
    kernel_cholesky_sequential(N, A);
    double serial_time = omp_get_wtime() - start;
    
    // Verify sequential result
    double error = verify_cholesky(N, A_orig, A);
    printf("Sequential: %.4f seconds (%.2f GFLOPS) [Error: %.2e]\n\n", 
           serial_time, total_flops / (serial_time * 1e9), error);
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    int num_thread_configs = 4;
    
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "Strategy", "Threads", "Time (s)", "Speedup", "Efficiency", "GFLOPS");
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "--------", "-------", "--------", "-------", "----------", "------");
    
    // Define strategies
    typedef void (*strategy_func)(int, double*);
    
    struct {
        const char* name;
        strategy_func func;
    } strategies[] = {
        {"Column-parallel", kernel_cholesky_column_parallel},
        {"Right-looking Blocked", kernel_cholesky_right_looking_blocked},
        {"Left-looking", kernel_cholesky_left_looking},
        {"Task-based", kernel_cholesky_tasks},
        {"Recursive", kernel_cholesky_recursive}
    };
    
    // Test each strategy
    for (int s = 0; s < 5; s++) {
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);
            
            // Time the strategy
            double times[MEASUREMENT_ITERATIONS];
            for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
                memcpy(A, A_orig, N * N * sizeof(double));
                start = omp_get_wtime();
                strategies[s].func(N, A);
                times[iter] = omp_get_wtime() - start;
            }
            
            // Calculate average time
            double avg_time = 0.0;
            for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
                avg_time += times[i];
            }
            avg_time /= MEASUREMENT_ITERATIONS;
            
            // Verify correctness
            memcpy(A, A_orig, N * N * sizeof(double));
            strategies[s].func(N, A);
            error = verify_cholesky(N, A_orig, A);
            
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
    free(A_orig);
    free(A);
    
    return 0;
}