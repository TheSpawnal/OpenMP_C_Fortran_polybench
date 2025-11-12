/*
 * Jacobi-2D Stencil Computation Benchmark
 * Multiple parallelization strategies for performance comparison
 * Iterative 5-point stencil computation with convergence
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define TOLERANCE 1e-6
#define MAX_ITER 5000
#define STENCIL_COEF 0.2

// Strategy 1: Basic parallel with red-black ordering
void jacobi2d_redblack(int n, double *A, double *B, int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        // Red phase (checkerboard pattern)
        #pragma omp parallel for collapse(2) reduction(max:diff)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if ((i + j) % 2 == 0) { // Red points
                    double newval = STENCIL_COEF * (A[(i-1)*n + j] + A[(i+1)*n + j] + 
                                                    A[i*n + (j-1)] + A[i*n + (j+1)] + 
                                                    A[i*n + j]);
                    diff = fmax(diff, fabs(newval - B[i*n + j]));
                    B[i*n + j] = newval;
                }
            }
        }
        
        // Black phase
        #pragma omp parallel for collapse(2) reduction(max:diff)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if ((i + j) % 2 == 1) { // Black points
                    double newval = STENCIL_COEF * (B[(i-1)*n + j] + B[(i+1)*n + j] + 
                                                    B[i*n + (j-1)] + B[i*n + (j+1)] + 
                                                    A[i*n + j]);
                    diff = fmax(diff, fabs(newval - B[i*n + j]));
                    B[i*n + j] = newval;
                }
            }
        }
        
        // Copy B back to A
        #pragma omp parallel for
        for (int i = 0; i < n * n; i++) {
            A[i] = B[i];
        }
        
        iter++;
    }
}

// Strategy 2: Wavefront parallelization (diagonal sweep)
void jacobi2d_wavefront(int n, double *A, double *B, int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        // Process diagonals
        for (int diag = 2; diag < 2 * n - 2; diag++) {
            #pragma omp parallel for reduction(max:diff)
            for (int i = 1; i < n - 1; i++) {
                int j = diag - i;
                if (j >= 1 && j < n - 1) {
                    double newval = STENCIL_COEF * (A[(i-1)*n + j] + A[(i+1)*n + j] + 
                                                    A[i*n + (j-1)] + A[i*n + (j+1)] + 
                                                    A[i*n + j]);
                    diff = fmax(diff, fabs(newval - B[i*n + j]));
                    B[i*n + j] = newval;
                }
            }
        }
        
        // Swap arrays
        double *temp = A;
        A = B;
        B = temp;
        
        iter++;
    }
}

// Strategy 3: Tiled/Blocked computation with ghost zones
void jacobi2d_tiled(int n, double *A, double *B, int max_iter, double tol, int tile_size) {
    int iter = 0;
    double diff = 1.0;
    
    int num_tiles = (n - 2 + tile_size - 1) / tile_size;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        #pragma omp parallel for collapse(2) reduction(max:diff)
        for (int ti = 0; ti < num_tiles; ti++) {
            for (int tj = 0; tj < num_tiles; tj++) {
                int i_start = ti * tile_size + 1;
                int j_start = tj * tile_size + 1;
                int i_end = (i_start + tile_size < n - 1) ? i_start + tile_size : n - 1;
                int j_end = (j_start + tile_size < n - 1) ? j_start + tile_size : n - 1;
                
                for (int i = i_start; i < i_end; i++) {
                    for (int j = j_start; j < j_end; j++) {
                        double newval = STENCIL_COEF * (A[(i-1)*n + j] + A[(i+1)*n + j] + 
                                                        A[i*n + (j-1)] + A[i*n + (j+1)] + 
                                                        A[i*n + j]);
                        diff = fmax(diff, fabs(newval - B[i*n + j]));
                        B[i*n + j] = newval;
                    }
                }
            }
        }
        
        // Swap arrays
        memcpy(A, B, n * n * sizeof(double));
        
        iter++;
    }
}

// Strategy 4: SIMD vectorization with aligned memory
void jacobi2d_simd(int n, double *A, double *B, int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    
    // Ensure aligned allocation for SIMD
    double *A_aligned = (double *)aligned_alloc(64, n * n * sizeof(double));
    double *B_aligned = (double *)aligned_alloc(64, n * n * sizeof(double));
    memcpy(A_aligned, A, n * n * sizeof(double));
    memcpy(B_aligned, B, n * n * sizeof(double));
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        #pragma omp parallel for reduction(max:diff)
        for (int i = 1; i < n - 1; i++) {
            double local_diff = 0.0;
            #pragma omp simd reduction(max:local_diff)
            for (int j = 1; j < n - 1; j++) {
                double newval = STENCIL_COEF * (A_aligned[(i-1)*n + j] + 
                                               A_aligned[(i+1)*n + j] + 
                                               A_aligned[i*n + (j-1)] + 
                                               A_aligned[i*n + (j+1)] + 
                                               A_aligned[i*n + j]);
                local_diff = fmax(local_diff, fabs(newval - B_aligned[i*n + j]));
                B_aligned[i*n + j] = newval;
            }
            diff = fmax(diff, local_diff);
        }
        
        // Swap pointers
        double *temp = A_aligned;
        A_aligned = B_aligned;
        B_aligned = temp;
        
        iter++;
    }
    
    // Copy result back
    memcpy(B, A_aligned, n * n * sizeof(double));
    
    free(A_aligned);
    free(B_aligned);
}

// Strategy 5: Hierarchical parallelization (nested parallel regions)
void jacobi2d_hierarchical(int n, double *A, double *B, int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    
    int chunk_rows = n / omp_get_max_threads();
    if (chunk_rows < 10) chunk_rows = 10;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        #pragma omp parallel reduction(max:diff)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            int rows_per_thread = (n - 2) / nthreads;
            int start_row = tid * rows_per_thread + 1;
            int end_row = (tid == nthreads - 1) ? n - 1 : start_row + rows_per_thread;
            
            double local_diff = 0.0;
            
            // Each thread processes its chunk
            for (int i = start_row; i < end_row; i++) {
                // Inner loop with SIMD
                #pragma omp simd reduction(max:local_diff)
                for (int j = 1; j < n - 1; j++) {
                    double newval = STENCIL_COEF * (A[(i-1)*n + j] + A[(i+1)*n + j] + 
                                                    A[i*n + (j-1)] + A[i*n + (j+1)] + 
                                                    A[i*n + j]);
                    local_diff = fmax(local_diff, fabs(newval - B[i*n + j]));
                    B[i*n + j] = newval;
                }
            }
            
            diff = fmax(diff, local_diff);
            
            // Synchronize before swap
            #pragma omp barrier
            #pragma omp for
            for (int i = 0; i < n * n; i++) {
                A[i] = B[i];
            }
        }
        
        iter++;
    }
}

// Strategy 6: Cache-optimized with temporal blocking
void jacobi2d_temporal_blocking(int n, double *A, double *B, int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    int time_block = 4; // Number of iterations to block
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        // Perform multiple iterations in temporal blocks
        for (int t = 0; t < time_block && iter < max_iter; t++) {
            #pragma omp parallel for collapse(2) reduction(max:diff)
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    double newval = STENCIL_COEF * (A[(i-1)*n + j] + A[(i+1)*n + j] + 
                                                    A[i*n + (j-1)] + A[i*n + (j+1)] + 
                                                    A[i*n + j]);
                    diff = fmax(diff, fabs(newval - B[i*n + j]));
                    B[i*n + j] = newval;
                }
            }
            
            // Swap for next iteration
            double *temp = A;
            A = B;
            B = temp;
            iter++;
        }
    }
}

// Initialize grid with boundary conditions
void init_grid(int n, double *A, double *B) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                // Boundary conditions
                A[i * n + j] = 100.0 * (1.0 + i + j) / (n + n);
                B[i * n + j] = A[i * n + j];
            } else {
                // Interior points
                A[i * n + j] = 0.0;
                B[i * n + j] = 0.0;
            }
        }
    }
}

// Compute residual for verification
double compute_residual(int n, double *A) {
    double residual = 0.0;
    
    #pragma omp parallel for reduction(+:residual)
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            double expected = STENCIL_COEF * (A[(i-1)*n + j] + A[(i+1)*n + j] + 
                                             A[i*n + (j-1)] + A[i*n + (j+1)] + 
                                             A[i*n + j]);
            residual += fabs(A[i*n + j] - expected);
        }
    }
    
    return residual / ((n - 2) * (n - 2));
}

int main(int argc, char **argv) {
    int n = 1024;
    int max_iter = 1000;
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    
    printf("Jacobi-2D Stencil Benchmark: n=%d, max_iter=%d\n", n, max_iter);
    printf("Threads: %d\n", omp_get_max_threads());
    
    // Allocate grids
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *A_orig = (double *)malloc(n * n * sizeof(double));
    
    double start_time, end_time, residual;
    
    // Strategy 1: Red-Black ordering
    init_grid(n, A, B);
    memcpy(A_orig, A, n * n * sizeof(double));
    start_time = omp_get_wtime();
    jacobi2d_redblack(n, A, B, max_iter, TOLERANCE);
    end_time = omp_get_wtime();
    residual = compute_residual(n, B);
    printf("Red-Black: %.4f seconds (residual: %.2e)\n", 
           end_time - start_time, residual);
    
    // Strategy 2: Wavefront
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = omp_get_wtime();
    jacobi2d_wavefront(n, A, B, max_iter, TOLERANCE);
    end_time = omp_get_wtime();
    residual = compute_residual(n, B);
    printf("Wavefront: %.4f seconds (residual: %.2e)\n", 
           end_time - start_time, residual);
    
    // Strategy 3: Tiled
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = omp_get_wtime();
    jacobi2d_tiled(n, A, B, max_iter, TOLERANCE, 64);
    end_time = omp_get_wtime();
    residual = compute_residual(n, B);
    printf("Tiled (64x64): %.4f seconds (residual: %.2e)\n", 
           end_time - start_time, residual);
    
    // Strategy 4: SIMD
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = omp_get_wtime();
    jacobi2d_simd(n, A, B, max_iter, TOLERANCE);
    end_time = omp_get_wtime();
    residual = compute_residual(n, B);
    printf("SIMD Vectorized: %.4f seconds (residual: %.2e)\n", 
           end_time - start_time, residual);
    
    // Strategy 5: Hierarchical
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = omp_get_wtime();
    jacobi2d_hierarchical(n, A, B, max_iter, TOLERANCE);
    end_time = omp_get_wtime();
    residual = compute_residual(n, B);
    printf("Hierarchical: %.4f seconds (residual: %.2e)\n", 
           end_time - start_time, residual);
    
    // Strategy 6: Temporal blocking
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = omp_get_wtime();
    jacobi2d_temporal_blocking(n, A, B, max_iter, TOLERANCE);
    end_time = omp_get_wtime();
    residual = compute_residual(n, B);
    printf("Temporal Blocking: %.4f seconds (residual: %.2e)\n", 
           end_time - start_time, residual);
    
    free(A);
    free(B);
    free(A_orig);
    
    return 0;
}
