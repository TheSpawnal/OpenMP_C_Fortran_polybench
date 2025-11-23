/*
 * Jacobi-2D Stencil Benchmark - IMPROVED VERSION
 * Enhanced with best practices from project knowledge base
 * Optimizations: improved convergence checking, better memory patterns, reduced synchronization
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

#define TOLERANCE 1e-6
#define MAX_ITER 5000
#define STENCIL_COEF 0.2
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

// Strategy 1: Optimized Red-Black with pointer swapping
void jacobi2d_redblack_opt(int n, double * __restrict__ A, 
                           double * __restrict__ B, 
                           int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    double *current = A;
    double *next = B;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        // Red phase - optimized with better memory access
        #pragma omp parallel for collapse(2) reduction(max:diff) schedule(static)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1 + (i & 1); j < n - 1; j += 2) {
                double newval = STENCIL_COEF * (
                    current[(i-1)*n + j] + current[(i+1)*n + j] + 
                    current[i*n + (j-1)] + current[i*n + (j+1)] + 
                    current[i*n + j]);
                diff = fmax(diff, fabs(newval - next[i*n + j]));
                next[i*n + j] = newval;
            }
        }
        
        // Black phase
        #pragma omp parallel for collapse(2) reduction(max:diff) schedule(static)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1 + ((i + 1) & 1); j < n - 1; j += 2) {
                double newval = STENCIL_COEF * (
                    next[(i-1)*n + j] + next[(i+1)*n + j] + 
                    next[i*n + (j-1)] + next[i*n + (j+1)] + 
                    current[i*n + j]);
                diff = fmax(diff, fabs(newval - next[i*n + j]));
                next[i*n + j] = newval;
            }
        }
        
        // Swap pointers instead of copying
        double *temp = current;
        current = next;
        next = temp;
        
        iter++;
    }
    
    // Ensure result is in B
    if (current != B) {
        memcpy(B, current, n * n * sizeof(double));
    }
}

// Strategy 2: Cache-blocked with temporal locality
void jacobi2d_blocked_temporal(int n, double * __restrict__ A,
                               double * __restrict__ B,
                               int max_iter, double tol, 
                               int block_size, int time_block) {
    int iter = 0;
    double diff = 1.0;
    double *arr[2] = {A, B};
    int current = 0;
    
    while (iter < max_iter && diff > tol) {
        // Perform multiple iterations in temporal blocks
        for (int t = 0; t < time_block && iter < max_iter && diff > tol; t++) {
            diff = 0.0;
            int next = 1 - current;
            
            #pragma omp parallel reduction(max:diff)
            {
                double local_diff = 0.0;
                
                #pragma omp for collapse(2) schedule(static) nowait
                for (int bi = 0; bi < n - 2; bi += block_size) {
                    for (int bj = 0; bj < n - 2; bj += block_size) {
                        int i_end = (bi + block_size < n - 1) ? bi + block_size : n - 1;
                        int j_end = (bj + block_size < n - 1) ? bj + block_size : n - 1;
                        
                        for (int i = bi + 1; i < i_end; i++) {
                            #pragma omp simd reduction(max:local_diff)
                            for (int j = bj + 1; j < j_end; j++) {
                                double newval = STENCIL_COEF * (
                                    arr[current][(i-1)*n + j] + 
                                    arr[current][(i+1)*n + j] + 
                                    arr[current][i*n + (j-1)] + 
                                    arr[current][i*n + (j+1)] + 
                                    arr[current][i*n + j]);
                                local_diff = fmax(local_diff, 
                                    fabs(newval - arr[next][i*n + j]));
                                arr[next][i*n + j] = newval;
                            }
                        }
                    }
                }
                diff = fmax(diff, local_diff);
            }
            
            current = next;
            iter++;
        }
    }
    
    // Ensure result is in B
    if (arr[current] != B) {
        memcpy(B, arr[current], n * n * sizeof(double));
    }
}

// Strategy 3: Wavefront with optimized diagonal processing
void jacobi2d_wavefront_opt(int n, double * __restrict__ A,
                            double * __restrict__ B,
                            int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    double *current = A;
    double *next = B;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        // Process anti-diagonals for better parallelism
        for (int diag = 2; diag < 2 * n - 2; diag++) {
            int i_start = fmax(1, diag - n + 2);
            int i_end = fmin(n - 1, diag - 1);
            int diag_len = i_end - i_start + 1;
            
            #pragma omp parallel for reduction(max:diff) schedule(static) if(diag_len > 16)
            for (int i = i_start; i < i_end; i++) {
                int j = diag - i;
                if (j >= 1 && j < n - 1) {
                    double newval = STENCIL_COEF * (
                        current[(i-1)*n + j] + current[(i+1)*n + j] + 
                        current[i*n + (j-1)] + current[i*n + (j+1)] + 
                        current[i*n + j]);
                    diff = fmax(diff, fabs(newval - next[i*n + j]));
                    next[i*n + j] = newval;
                }
            }
        }
        
        // Swap arrays
        double *temp = current;
        current = next;
        next = temp;
        
        iter++;
    }
    
    // Ensure result is in B
    if (current != B) {
        memcpy(B, current, n * n * sizeof(double));
    }
}

// Strategy 4: SIMD-optimized with aligned access
void jacobi2d_simd_aligned(int n, double * __restrict__ A,
                           double * __restrict__ B,
                           int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    
    // Ensure aligned pointers
    __assume_aligned(A, ALIGN_SIZE);
    __assume_aligned(B, ALIGN_SIZE);
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        #pragma omp parallel reduction(max:diff)
        {
            double local_diff = 0.0;
            
            #pragma omp for schedule(static) nowait
            for (int i = 1; i < n - 1; i++) {
                // Process row with SIMD
                #pragma omp simd reduction(max:local_diff) aligned(A,B:ALIGN_SIZE)
                for (int j = 1; j < n - 1; j++) {
                    double newval = STENCIL_COEF * (
                        A[(i-1)*n + j] + A[(i+1)*n + j] + 
                        A[i*n + (j-1)] + A[i*n + (j+1)] + 
                        A[i*n + j]);
                    local_diff = fmax(local_diff, fabs(newval - B[i*n + j]));
                    B[i*n + j] = newval;
                }
            }
            diff = fmax(diff, local_diff);
        }
        
        // Swap pointers
        double *temp = A;
        A = B;
        B = temp;
        
        iter++;
    }
}

// Strategy 5: Task-based with dependency management
void jacobi2d_tasks_deps(int n, double * __restrict__ A,
                        double * __restrict__ B,
                        int max_iter, double tol, int tile_size) {
    int iter = 0;
    double diff = 1.0;
    double *current = A;
    double *next = B;
    
    int num_tiles = (n - 2 + tile_size - 1) / tile_size;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int ti = 0; ti < num_tiles; ti++) {
                    for (int tj = 0; tj < num_tiles; tj++) {
                        #pragma omp task depend(in:current[ti*tile_size*n:tile_size*n]) \
                                         depend(out:next[ti*tile_size*n:tile_size*n]) \
                                         shared(diff)
                        {
                            int i_start = ti * tile_size + 1;
                            int j_start = tj * tile_size + 1;
                            int i_end = fmin(i_start + tile_size, n - 1);
                            int j_end = fmin(j_start + tile_size, n - 1);
                            
                            double local_diff = 0.0;
                            for (int i = i_start; i < i_end; i++) {
                                for (int j = j_start; j < j_end; j++) {
                                    double newval = STENCIL_COEF * (
                                        current[(i-1)*n + j] + current[(i+1)*n + j] + 
                                        current[i*n + (j-1)] + current[i*n + (j+1)] + 
                                        current[i*n + j]);
                                    local_diff = fmax(local_diff, 
                                        fabs(newval - next[i*n + j]));
                                    next[i*n + j] = newval;
                                }
                            }
                            
                            #pragma omp atomic
                            diff = fmax(diff, local_diff);
                        }
                    }
                }
            }
        }
        
        // Swap arrays
        double *temp = current;
        current = next;
        next = temp;
        
        iter++;
    }
    
    // Ensure result is in B
    if (current != B) {
        memcpy(B, current, n * n * sizeof(double));
    }
}

// Strategy 6: Hierarchical with nested parallelism
void jacobi2d_hierarchical_opt(int n, double * __restrict__ A,
                               double * __restrict__ B,
                               int max_iter, double tol) {
    int iter = 0;
    double diff = 1.0;
    double *current = A;
    double *next = B;
    
    int outer_threads = omp_get_max_threads();
    int chunk_rows = (n - 2) / outer_threads;
    if (chunk_rows < 16) chunk_rows = 16;
    
    while (iter < max_iter && diff > tol) {
        diff = 0.0;
        
        #pragma omp parallel reduction(max:diff)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            int start_row = tid * chunk_rows + 1;
            int end_row = (tid == nthreads - 1) ? n - 1 : start_row + chunk_rows;
            if (end_row > n - 1) end_row = n - 1;
            
            double local_diff = 0.0;
            
            for (int i = start_row; i < end_row; i++) {
                #pragma omp simd reduction(max:local_diff)
                for (int j = 1; j < n - 1; j++) {
                    double newval = STENCIL_COEF * (
                        current[(i-1)*n + j] + current[(i+1)*n + j] + 
                        current[i*n + (j-1)] + current[i*n + (j+1)] + 
                        current[i*n + j]);
                    local_diff = fmax(local_diff, fabs(newval - next[i*n + j]));
                    next[i*n + j] = newval;
                }
            }
            
            diff = fmax(diff, local_diff);
        }
        
        // Swap pointers
        double *temp = current;
        current = next;
        next = temp;
        
        iter++;
    }
    
    // Ensure result is in B
    if (current != B) {
        memcpy(B, current, n * n * sizeof(double));
    }
}

// Initialize grid with boundary conditions
void init_grid(int n, double *A, double *B) {
    unsigned int seed = 42;
    
    #pragma omp parallel for firstprivate(seed)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                // Boundary conditions
                seed = seed * 1103515245 + 12345 + i * n + j;
                A[i * n + j] = 100.0 * (1.0 + (seed % 100) / 100.0);
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
            double expected = STENCIL_COEF * (
                A[(i-1)*n + j] + A[(i+1)*n + j] + 
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
    int nthreads = omp_get_max_threads();
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    if (argc > 3) {
        nthreads = atoi(argv[3]);
        omp_set_num_threads(nthreads);
    }
    
    printf("===== Improved Jacobi-2D Stencil Benchmark =====\n");
    printf("Grid size: %d x %d\n", n, n);
    printf("Max iterations: %d\n", max_iter);
    printf("Threads: %d\n", nthreads);
    printf("Memory: %.2f MB\n\n", (2 * n * n * sizeof(double)) / (1024.0 * 1024.0));
    
    // Allocate aligned grids
    double *A = (double *)alloc_aligned(n * n * sizeof(double));
    double *B = (double *)alloc_aligned(n * n * sizeof(double));
    double *A_orig = (double *)alloc_aligned(n * n * sizeof(double));
    
    // Initialize grid
    init_grid(n, A_orig, B);
    
    double start_time, end_time, residual;
    double best_time = 1e9;
    const char *best_method = "";
    double flops = 6.0 * (n - 2) * (n - 2) * max_iter;  // 6 ops per point per iteration
    
    // Warmup
    printf("Warming up...\n");
    memcpy(A, A_orig, n * n * sizeof(double));
    for (int w = 0; w < WARMUP_ITER; w++) {
        jacobi2d_simd_aligned(n, A, B, 10, TOLERANCE);
    }
    
    printf("Performance Results:\n");
    printf("%-25s %10s %12s %10s\n", "Method", "Time (s)", "GFLOP/s", "Residual");
    printf("%-25s %10s %12s %10s\n", "------", "--------", "-------", "--------");
    
    // Test Strategy 1: Red-Black optimized
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = wtime();
    jacobi2d_redblack_opt(n, A, B, max_iter, TOLERANCE);
    end_time = wtime();
    double time1 = end_time - start_time;
    residual = compute_residual(n, B);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Red-Black Optimized", time1, (flops/1e9)/time1, residual);
    if (time1 < best_time) { best_time = time1; best_method = "Red-Black"; }
    
    // Test Strategy 2: Blocked temporal
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = wtime();
    jacobi2d_blocked_temporal(n, A, B, max_iter, TOLERANCE, 64, 4);
    end_time = wtime();
    double time2 = end_time - start_time;
    residual = compute_residual(n, B);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Blocked Temporal (64,4)", time2, (flops/1e9)/time2, residual);
    if (time2 < best_time) { best_time = time2; best_method = "Blocked Temporal"; }
    
    // Test Strategy 3: Wavefront optimized
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = wtime();
    jacobi2d_wavefront_opt(n, A, B, max_iter, TOLERANCE);
    end_time = wtime();
    double time3 = end_time - start_time;
    residual = compute_residual(n, B);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Wavefront Optimized", time3, (flops/1e9)/time3, residual);
    if (time3 < best_time) { best_time = time3; best_method = "Wavefront"; }
    
    // Test Strategy 4: SIMD aligned
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = wtime();
    jacobi2d_simd_aligned(n, A, B, max_iter, TOLERANCE);
    end_time = wtime();
    double time4 = end_time - start_time;
    residual = compute_residual(n, B);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "SIMD Aligned", time4, (flops/1e9)/time4, residual);
    if (time4 < best_time) { best_time = time4; best_method = "SIMD"; }
    
    // Test Strategy 5: Tasks with dependencies
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = wtime();
    jacobi2d_tasks_deps(n, A, B, max_iter, TOLERANCE, 64);
    end_time = wtime();
    double time5 = end_time - start_time;
    residual = compute_residual(n, B);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Tasks with Deps (64)", time5, (flops/1e9)/time5, residual);
    if (time5 < best_time) { best_time = time5; best_method = "Tasks"; }
    
    // Test Strategy 6: Hierarchical
    memcpy(A, A_orig, n * n * sizeof(double));
    init_grid(n, A, B);
    start_time = wtime();
    jacobi2d_hierarchical_opt(n, A, B, max_iter, TOLERANCE);
    end_time = wtime();
    double time6 = end_time - start_time;
    residual = compute_residual(n, B);
    printf("%-25s %10.4f %12.2f %10.2e\n", 
           "Hierarchical Optimized", time6, (flops/1e9)/time6, residual);
    if (time6 < best_time) { best_time = time6; best_method = "Hierarchical"; }
    
    printf("\nBest method: %s (%.4f seconds)\n", best_method, best_time);
    printf("Best performance: %.2f GFLOP/s\n", (flops/1e9)/best_time);
    printf("Speedup vs baseline: %.2fx\n", time1 / best_time);
    
    free(A);
    free(B);
    free(A_orig);
    
    return 0;
}
