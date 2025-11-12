/*
 * Cholesky Decomposition Benchmark
 * Multiple parallelization strategies for performance comparison
 * Decomposes a positive definite matrix A into L*L^T
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define TOL 1e-6

// Strategy 1: Column-wise parallelization
void cholesky_column_parallel(int n, double *A, double *L) {
    // Copy A to L
    memcpy(L, A, n * n * sizeof(double));
    
    for (int j = 0; j < n; j++) {
        // Compute diagonal element
        double sum = 0.0;
        for (int k = 0; k < j; k++) {
            sum += L[j * n + k] * L[j * n + k];
        }
        L[j * n + j] = sqrt(L[j * n + j] - sum);
        
        // Update column j below diagonal (parallel)
        #pragma omp parallel for schedule(dynamic)
        for (int i = j + 1; i < n; i++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) {
                s += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (L[i * n + j] - s) / L[j * n + j];
        }
        
        // Zero out upper triangle
        #pragma omp parallel for
        for (int i = 0; i < j; i++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 2: Block-wise Cholesky with tasks
void cholesky_blocked_tasks(int n, double *A, double *L, int block_size) {
    memcpy(L, A, n * n * sizeof(double));
    
    int nb = (n + block_size - 1) / block_size;
    
    #pragma omp parallel
    #pragma omp single
    {
        for (int k = 0; k < nb; k++) {
            int k_start = k * block_size;
            int k_end = (k_start + block_size < n) ? k_start + block_size : n;
            
            // Diagonal block factorization (sequential)
            #pragma omp task firstprivate(k_start, k_end)
            {
                for (int j = k_start; j < k_end; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < j; p++) {
                        sum += L[j * n + p] * L[j * n + p];
                    }
                    L[j * n + j] = sqrt(L[j * n + j] - sum);
                    
                    for (int i = j + 1; i < k_end; i++) {
                        double s = 0.0;
                        for (int p = 0; p < j; p++) {
                            s += L[i * n + p] * L[j * n + p];
                        }
                        L[i * n + j] = (L[i * n + j] - s) / L[j * n + j];
                    }
                }
            }
            #pragma omp taskwait
            
            // Panel update (parallel tasks)
            for (int i = k + 1; i < nb; i++) {
                int i_start = i * block_size;
                int i_end = (i_start + block_size < n) ? i_start + block_size : n;
                
                #pragma omp task firstprivate(i_start, i_end, k_start, k_end)
                {
                    for (int jj = k_start; jj < k_end; jj++) {
                        for (int ii = i_start; ii < i_end; ii++) {
                            double s = 0.0;
                            for (int p = 0; p < jj; p++) {
                                s += L[ii * n + p] * L[jj * n + p];
                            }
                            L[ii * n + jj] = (L[ii * n + jj] - s) / L[jj * n + jj];
                        }
                    }
                }
            }
            #pragma omp taskwait
            
            // Trailing matrix update (parallel tasks)
            for (int i = k + 1; i < nb; i++) {
                for (int j = k + 1; j <= i; j++) {
                    int i_start = i * block_size;
                    int i_end = (i_start + block_size < n) ? i_start + block_size : n;
                    int j_start = j * block_size;
                    int j_end = (j_start + block_size < n) ? j_start + block_size : n;
                    
                    #pragma omp task firstprivate(i_start, i_end, j_start, j_end, k_start, k_end)
                    {
                        for (int ii = i_start; ii < i_end; ii++) {
                            for (int jj = j_start; jj < j_end && jj <= ii; jj++) {
                                for (int kk = k_start; kk < k_end; kk++) {
                                    L[ii * n + jj] -= L[ii * n + kk] * L[jj * n + kk];
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
        
        // Zero out upper triangle
        #pragma omp taskloop
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                L[i * n + j] = 0.0;
            }
        }
    }
}

// Strategy 3: Right-looking Cholesky with OpenMP sections
void cholesky_right_looking(int n, double *A, double *L) {
    memcpy(L, A, n * n * sizeof(double));
    
    for (int k = 0; k < n; k++) {
        // Compute L[k][k]
        L[k * n + k] = sqrt(L[k * n + k]);
        
        // Scale column k
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            L[i * n + k] /= L[k * n + k];
        }
        
        // Update trailing submatrix
        #pragma omp parallel for collapse(2)
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                L[i * n + j] -= L[i * n + k] * L[j * n + k];
            }
        }
    }
    
    // Zero out upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 4: Left-looking Cholesky with fine-grained synchronization
void cholesky_left_looking(int n, double *A, double *L) {
    memcpy(L, A, n * n * sizeof(double));
    
    for (int j = 0; j < n; j++) {
        // Update column j using previous columns
        #pragma omp parallel for schedule(static)
        for (int i = j; i < n; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] -= sum;
        }
        
        // Compute diagonal element
        L[j * n + j] = sqrt(L[j * n + j]);
        
        // Scale column j
        #pragma omp parallel for
        for (int i = j + 1; i < n; i++) {
            L[i * n + j] /= L[j * n + j];
        }
    }
    
    // Zero out upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Strategy 5: Recursive Cholesky with nested parallelism
void cholesky_recursive_helper(int n, double *A, int lda, int depth) {
    if (n <= 64) {
        // Base case: sequential Cholesky for small matrices
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += A[j * lda + k] * A[j * lda + k];
            }
            A[j * lda + j] = sqrt(A[j * lda + j] - sum);
            
            for (int i = j + 1; i < n; i++) {
                double s = 0.0;
                for (int k = 0; k < j; k++) {
                    s += A[i * lda + k] * A[j * lda + k];
                }
                A[i * lda + j] = (A[i * lda + j] - s) / A[j * lda + j];
            }
        }
        return;
    }
    
    int n1 = n / 2;
    int n2 = n - n1;
    
    // Cholesky of A11
    cholesky_recursive_helper(n1, A, lda, depth + 1);
    
    // Solve for A21
    #pragma omp parallel for if(depth < 2)
    for (int i = n1; i < n; i++) {
        for (int j = 0; j < n1; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += A[i * lda + k] * A[j * lda + k];
            }
            A[i * lda + j] = (A[i * lda + j] - sum) / A[j * lda + j];
        }
    }
    
    // Update A22
    #pragma omp parallel for collapse(2) if(depth < 2)
    for (int i = n1; i < n; i++) {
        for (int j = n1; j <= i; j++) {
            for (int k = 0; k < n1; k++) {
                A[i * lda + j] -= A[i * lda + k] * A[j * lda + k];
            }
        }
    }
    
    // Cholesky of A22
    cholesky_recursive_helper(n2, &A[n1 * lda + n1], lda, depth + 1);
}

void cholesky_recursive(int n, double *A, double *L) {
    memcpy(L, A, n * n * sizeof(double));
    cholesky_recursive_helper(n, L, n, 0);
    
    // Zero out upper triangle
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            L[i * n + j] = 0.0;
        }
    }
}

// Generate a positive definite matrix
void generate_posdef_matrix(int n, double *A) {
    // Create a random lower triangular matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i >= j) {
                A[i * n + j] = (double)rand() / RAND_MAX;
            } else {
                A[i * n + j] = 0.0;
            }
        }
        A[i * n + i] += n; // Make diagonally dominant
    }
    
    // Compute A = L * L^T
    double *temp = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += A[i * n + k] * A[j * n + k];
            }
            temp[i * n + j] = sum;
            temp[j * n + i] = sum;
        }
    }
    memcpy(A, temp, n * n * sizeof(double));
    free(temp);
}

// Verify Cholesky decomposition
double verify_cholesky(int n, double *A, double *L) {
    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            double error = fabs(A[i * n + j] - sum);
            if (error > max_error) max_error = error;
        }
    }
    return max_error;
}

int main(int argc, char **argv) {
    int n = 1500;
    if (argc > 1) n = atoi(argv[1]);
    
    printf("Cholesky Decomposition Benchmark: n=%d\n", n);
    printf("Threads: %d\n", omp_get_max_threads());
    
    // Allocate matrices
    double *A = (double *)malloc(n * n * sizeof(double));
    double *L = (double *)malloc(n * n * sizeof(double));
    
    // Generate positive definite matrix
    generate_posdef_matrix(n, A);
    
    double start_time, end_time, error;
    
    // Strategy 1: Column-wise parallel
    start_time = omp_get_wtime();
    cholesky_column_parallel(n, A, L);
    end_time = omp_get_wtime();
    error = verify_cholesky(n, A, L);
    printf("Column-wise Parallel: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 2: Blocked with tasks
    start_time = omp_get_wtime();
    cholesky_blocked_tasks(n, A, L, 128);
    end_time = omp_get_wtime();
    error = verify_cholesky(n, A, L);
    printf("Blocked Tasks (128): %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 3: Right-looking
    start_time = omp_get_wtime();
    cholesky_right_looking(n, A, L);
    end_time = omp_get_wtime();
    error = verify_cholesky(n, A, L);
    printf("Right-looking: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 4: Left-looking
    start_time = omp_get_wtime();
    cholesky_left_looking(n, A, L);
    end_time = omp_get_wtime();
    error = verify_cholesky(n, A, L);
    printf("Left-looking: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 5: Recursive
    start_time = omp_get_wtime();
    cholesky_recursive(n, A, L);
    end_time = omp_get_wtime();
    error = verify_cholesky(n, A, L);
    printf("Recursive: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    free(A);
    free(L);
    
    return 0;
}
