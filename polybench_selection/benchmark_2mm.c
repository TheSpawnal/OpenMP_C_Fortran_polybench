/*
 * 2mm Benchmark - Double Matrix Multiplication: D = alpha*A*B*C + beta*D
 * Multiple parallelization strategies for performance comparison

 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#define ALPHA 1.5
#define BETA 1.2
#define TOL 0.001

// Strategy 1: Basic parallel for with different loop orderings
void mm2_basic_parallel(int ni, int nj, int nk, int nl,
                        double *A, double *B, double *C, double *D,
                        double *tmp, double alpha, double beta) {
    // First multiplication: tmp = A * B
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[i * nj + j] = 0.0;
            for (int k = 0; k < nk; k++) {
                tmp[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
            }
        }
    }
    
    // Second multiplication: D = tmp * C + beta*D
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i * nl + j];
            for (int k = 0; k < nj; k++) {
                sum += tmp[i * nj + k] * C[k * nl + j];
            }
            D[i * nl + j] = sum;
        }
    }
}

// Strategy 2: Collapsed loops with better cache locality
void mm2_collapsed(int ni, int nj, int nk, int nl,
                   double *A, double *B, double *C, double *D,
                   double *tmp, double alpha, double beta) {
    // First multiplication with collapsed loops
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            for (int k = 0; k < nk; k++) {
                sum += alpha * A[i * nk + k] * B[k * nj + j];
            }
            tmp[i * nj + j] = sum;
        }
    }
    
    // Second multiplication with collapsed loops
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            double sum = beta * D[i * nl + j];
            for (int k = 0; k < nj; k++) {
                sum += tmp[i * nj + k] * C[k * nl + j];
            }
            D[i * nl + j] = sum;
        }
    }
}

// Strategy 3: Tiled/Blocked multiplication for better cache usage
void mm2_tiled(int ni, int nj, int nk, int nl,
               double *A, double *B, double *C, double *D,
               double *tmp, double alpha, double beta, int tile_size) {
    
    // Initialize tmp
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            tmp[i * nj + j] = 0.0;
        }
    }
    
    // First multiplication with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += tile_size) {
        for (int jj = 0; jj < nj; jj += tile_size) {
            for (int kk = 0; kk < nk; kk += tile_size) {
                for (int i = ii; i < ii + tile_size && i < ni; i++) {
                    for (int j = jj; j < jj + tile_size && j < nj; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < kk + tile_size && k < nk; k++) {
                            sum += alpha * A[i * nk + k] * B[k * nj + j];
                        }
                        tmp[i * nj + j] += sum;
                    }
                }
            }
        }
    }
    
    // Second multiplication with tiling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += tile_size) {
        for (int jj = 0; jj < nl; jj += tile_size) {
            for (int i = ii; i < ii + tile_size && i < ni; i++) {
                for (int j = jj; j < jj + tile_size && j < nl; j++) {
                    double sum = beta * D[i * nl + j];
                    for (int kk = 0; kk < nj; kk += tile_size) {
                        for (int k = kk; k < kk + tile_size && k < nj; k++) {
                            sum += tmp[i * nj + k] * C[k * nl + j];
                        }
                    }
                    D[i * nl + j] = sum;
                }
            }
        }
    }
}

// Strategy 4: SIMD vectorization with OpenMP
void mm2_simd(int ni, int nj, int nk, int nl,
              double *A, double *B, double *C, double *D,
              double *tmp, double alpha, double beta) {
    
    // First multiplication with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < nk; k++) {
                sum += alpha * A[i * nk + k] * B[k * nj + j];
            }
            tmp[i * nj + j] = sum;
        }
    }
    
    // Second multiplication with SIMD
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
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

// Strategy 5: Task-based parallelism for load balancing
void mm2_tasks(int ni, int nj, int nk, int nl,
               double *A, double *B, double *C, double *D,
               double *tmp, double alpha, double beta, int chunk) {
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // First multiplication with tasks
            for (int i = 0; i < ni; i += chunk) {
                for (int j = 0; j < nj; j += chunk) {
                    #pragma omp task firstprivate(i, j)
                    {
                        int i_end = (i + chunk < ni) ? i + chunk : ni;
                        int j_end = (j + chunk < nj) ? j + chunk : nj;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = 0.0;
                                for (int k = 0; k < nk; k++) {
                                    sum += alpha * A[ii * nk + k] * B[k * nj + jj];
                                }
                                tmp[ii * nj + jj] = sum;
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
            
            // Second multiplication with tasks
            for (int i = 0; i < ni; i += chunk) {
                for (int j = 0; j < nl; j += chunk) {
                    #pragma omp task firstprivate(i, j)
                    {
                        int i_end = (i + chunk < ni) ? i + chunk : ni;
                        int j_end = (j + chunk < nl) ? j + chunk : nl;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            for (int jj = j; jj < j_end; jj++) {
                                double sum = beta * D[ii * nl + jj];
                                for (int k = 0; k < nj; k++) {
                                    sum += tmp[ii * nj + k] * C[k * nl + jj];
                                }
                                D[ii * nl + jj] = sum;
                              }
                        }
                    }
                }
            }
        }
    }
}

// Initialize matrices
void init_matrices(int ni, int nj, int nk, int nl,
                   double *A, double *B, double *C, double *D) {
    for (int i = 0; i < ni * nk; i++)
        A[i] = (double)(i % 100) / 100.0;
    for (int i = 0; i < nk * nj; i++)
        B[i] = (double)(i % 100) / 100.0;
    for (int i = 0; i < nj * nl; i++)
        C[i] = (double)(i % 100) / 100.0;
    for (int i = 0; i < ni * nl; i++)
        D[i] = (double)(i % 100) / 100.0;
}

int main(int argc, char **argv) {
    int ni = 800, nj = 900, nk = 1000, nl = 1100;
    
    if (argc > 1) ni = atoi(argv[1]);
    if (argc > 2) nj = atoi(argv[2]);
    if (argc > 3) nk = atoi(argv[3]);
    if (argc > 4) nl = atoi(argv[4]);
    
    printf("2mm Benchmark: ni=%d, nj=%d, nk=%d, nl=%d\n", ni, nj, nk, nl);
    printf("Threads: %d\n", omp_get_max_threads());
    
    //allocate matrices
    double *A = (double *)malloc(ni * nk * sizeof(double));
    double *B = (double *)malloc(nk * nj * sizeof(double));
    double *C = (double *)malloc(nj * nl * sizeof(double));
    double *D = (double *)malloc(ni * nl * sizeof(double));
    double *D_ref = (double *)malloc(ni * nl * sizeof(double));
    double *tmp = (double *)malloc(ni * nj * sizeof(double));
    
    double start_time, end_time;
    
    // Test Strategy 1:basic parallel
    init_matrices(ni, nj, nk, nl, A, B, C, D);
    memcpy(D_ref, D, ni * nl * sizeof(double));
    
    start_time = omp_get_wtime();
    mm2_basic_parallel(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    end_time = omp_get_wtime();
    printf("Basic Parallel: %.4f seconds\n", end_time - start_time);
    
    // Test Strategy 2: Collapsed loops
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = omp_get_wtime();
    mm2_collapsed(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    end_time = omp_get_wtime();
    printf("Collapsed Loops: %.4f seconds\n", end_time - start_time);
    
    // Test Strategy 3: Tiled
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = omp_get_wtime();
    mm2_tiled(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA, 64);
    end_time = omp_get_wtime();
    printf("Tiled (64x64): %.4f seconds\n", end_time - start_time);
    
    // Test Strategy 4: SIMD
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = omp_get_wtime();
    mm2_simd(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA);
    end_time = omp_get_wtime();
    printf("SIMD Vectorization: %.4f seconds\n", end_time - start_time);
    
    // Test Strategy 5: Tasks
    memcpy(D, D_ref, ni * nl * sizeof(double));
    start_time = omp_get_wtime();
    mm2_tasks(ni, nj, nk, nl, A, B, C, D, tmp, ALPHA, BETA, 100);
    end_time = omp_get_wtime();
    printf("Task-based: %.4f seconds\n", end_time - start_time);
    
    free(A); free(B); free(C); free(D); free(D_ref); free(tmp);
    
    return 0;
}
