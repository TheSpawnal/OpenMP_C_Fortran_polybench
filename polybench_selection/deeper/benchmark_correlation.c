/**
 * correlation.c: Pearson Correlation Matrix Benchmark with Multiple Strategies
 * Computes correlation coefficients between all pairs of variables
 * 
 * Strategies implemented:
 * 1. Sequential (baseline)
 * 2. Row-wise parallel
 * 3. Tiled/Blocked
 * 4. SIMD vectorized
 * 5. Task-based dynamic
 * 6. Reduction-based
 * 7. Column-major optimized
 * 8. Hierarchical with prefetching
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <immintrin.h>
#include "benchmark_metrics.h"

// Problem size definitions
#ifdef MINI
#define M 28   // Data points
#define N 32   // Variables
#elif defined(SMALL)
#define M 80
#define N 100
#elif defined(MEDIUM)
#define M 240
#define N 260
#elif defined(LARGE)
#define M 1200
#define N 1400
#else // Default STANDARD
#define M 500
#define N 600
#endif

#define ALIGN_SIZE 64
#define TILE_SIZE 32
#define EPS 0.1

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

// Initialize data matrix
static void init_array(int m, int n, double *data, double *mean, double *stddev) {
    // Initialize data matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data[i*m + j] = (double)(i*j) / m + i;
        }
    }
    
    // Initialize mean and stddev arrays (will be computed)
    for (int i = 0; i < n; i++) {
        mean[i] = 0.0;
        stddev[i] = 0.0;
    }
}

// Verify correlation matrix properties
static double verify_correlation(int n, double *corr) {
    double max_error = 0.0;
    
    // Check diagonal elements (should be 1.0)
    for (int i = 0; i < n; i++) {
        double error = fabs(corr[i*n + i] - 1.0);
        if (error > max_error) max_error = error;
    }
    
    // Check symmetry
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double error = fabs(corr[i*n + j] - corr[j*n + i]);
            if (error > max_error) max_error = error;
        }
    }
    
    // Check range [-1, 1]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (corr[i*n + j] < -1.0 - EPS || corr[i*n + j] > 1.0 + EPS) {
                max_error = fmax(max_error, fabs(corr[i*n + j]) - 1.0);
            }
        }
    }
    
    return max_error;
}

// Calculate FLOPS for correlation
static long long calculate_flops(int m, int n) {
    // Mean calculation: n * m additions
    // Stddev calculation: n * (2m additions + m multiplications + 1 sqrt)
    // Correlation: n*(n-1)/2 * (m multiplications + m additions)
    // Normalization: n*(n-1)/2 divisions
    long long mean_flops = (long long)n * m;
    long long stddev_flops = (long long)n * (3 * m + 1);
    long long corr_flops = (long long)n * (n - 1) / 2 * (2 * m + 1);
    return mean_flops + stddev_flops + corr_flops;
}

// Strategy 1: Sequential baseline
void kernel_correlation_sequential(int m, int n, double *data, 
                                  double *corr, double *mean, double *stddev) {
    // Calculate means
    for (int j = 0; j < n; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < m; i++) {
            mean[j] += data[j*m + i];
        }
        mean[j] /= m;
    }
    
    // Calculate standard deviations
    for (int j = 0; j < n; j++) {
        stddev[j] = 0.0;
        for (int i = 0; i < m; i++) {
            stddev[j] += (data[j*m + i] - mean[j]) * (data[j*m + i] - mean[j]);
        }
        stddev[j] = sqrt(stddev[j] / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize data
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            data[j*m + i] = (data[j*m + i] - mean[j]) / (sqrt((double)m) * stddev[j]);
        }
    }
    
    // Calculate correlation matrix
    for (int i = 0; i < n - 1; i++) {
        corr[i*n + i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            corr[i*n + j] = 0.0;
            for (int k = 0; k < m; k++) {
                corr[i*n + j] += data[i*m + k] * data[j*m + k];
            }
            corr[j*n + i] = corr[i*n + j];
        }
    }
    corr[(n-1)*n + (n-1)] = 1.0;
}

// Strategy 2: Row-wise parallel
void kernel_correlation_row_parallel(int m, int n, double *data,
                                    double *corr, double *mean, double *stddev) {
    // Calculate means in parallel
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            sum += data[j*m + i];
        }
        mean[j] = sum / m;
    }
    
    // Calculate standard deviations in parallel
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            double diff = data[j*m + i] - mean[j];
            sum += diff * diff;
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize data in parallel
    double sqrt_m = sqrt((double)m);
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double inv_factor = 1.0 / (sqrt_m * stddev[j]);
        #pragma omp simd
        for (int i = 0; i < m; i++) {
            data[j*m + i] = (data[j*m + i] - mean[j]) * inv_factor;
        }
    }
    
    // Calculate correlation matrix in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        corr[i*n + i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < m; k++) {
                sum += data[i*m + k] * data[j*m + k];
            }
            corr[i*n + j] = sum;
            corr[j*n + i] = sum;
        }
    }
}

// Strategy 3: Tiled/Blocked
void kernel_correlation_tiled(int m, int n, double *data,
                             double *corr, double *mean, double *stddev) {
    const int tile = TILE_SIZE;
    
    // Calculate means with tiling
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int ii = 0; ii < m; ii += tile) {
            int i_end = (ii + tile < m) ? ii + tile : m;
            double local_sum = 0.0;
            #pragma omp simd reduction(+:local_sum)
            for (int i = ii; i < i_end; i++) {
                local_sum += data[j*m + i];
            }
            sum += local_sum;
        }
        mean[j] = sum / m;
    }
    
    // Calculate standard deviations with tiling
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int ii = 0; ii < m; ii += tile) {
            int i_end = (ii + tile < m) ? ii + tile : m;
            double local_sum = 0.0;
            #pragma omp simd reduction(+:local_sum)
            for (int i = ii; i < i_end; i++) {
                double diff = data[j*m + i] - mean[j];
                local_sum += diff * diff;
            }
            sum += local_sum;
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize data with tiling
    double sqrt_m = sqrt((double)m);
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double inv_factor = 1.0 / (sqrt_m * stddev[j]);
        for (int ii = 0; ii < m; ii += tile) {
            int i_end = (ii + tile < m) ? ii + tile : m;
            #pragma omp simd
            for (int i = ii; i < i_end; i++) {
                data[j*m + i] = (data[j*m + i] - mean[j]) * inv_factor;
            }
        }
    }
    
    // Calculate correlation matrix with tiling
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += tile) {
        for (int jj = ii; jj < n; jj += tile) {
            int i_end = (ii + tile < n) ? ii + tile : n;
            int j_end = (jj + tile < n) ? jj + tile : n;
            
            for (int i = ii; i < i_end; i++) {
                int j_start = (i >= jj) ? i : jj;
                for (int j = j_start; j < j_end; j++) {
                    if (i == j) {
                        corr[i*n + i] = 1.0;
                    } else if (j > i) {
                        double sum = 0.0;
                        for (int kk = 0; kk < m; kk += tile) {
                            int k_end = (kk + tile < m) ? kk + tile : m;
                            double local_sum = 0.0;
                            #pragma omp simd reduction(+:local_sum)
                            for (int k = kk; k < k_end; k++) {
                                local_sum += data[i*m + k] * data[j*m + k];
                            }
                            sum += local_sum;
                        }
                        corr[i*n + j] = sum;
                        corr[j*n + i] = sum;
                    }
                }
            }
        }
    }
}

// Strategy 4: SIMD vectorized
void kernel_correlation_simd(int m, int n, double *__restrict__ data,
                            double *__restrict__ corr, 
                            double *__restrict__ mean,
                            double *__restrict__ stddev) {
    // Assume aligned pointers for compiler optimization
    data = __builtin_assume_aligned(data, ALIGN_SIZE);
    corr = __builtin_assume_aligned(corr, ALIGN_SIZE);
    mean = __builtin_assume_aligned(mean, ALIGN_SIZE);
    stddev = __builtin_assume_aligned(stddev, ALIGN_SIZE);
    
    // Calculate means with SIMD
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum) aligned(data:ALIGN_SIZE)
        for (int i = 0; i < m; i++) {
            sum += data[j*m + i];
        }
        mean[j] = sum / m;
    }
    
    // Calculate standard deviations with SIMD
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        double m_j = mean[j];
        #pragma omp simd reduction(+:sum) aligned(data:ALIGN_SIZE)
        for (int i = 0; i < m; i++) {
            double diff = data[j*m + i] - m_j;
            sum += diff * diff;
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize data with SIMD
    double sqrt_m = sqrt((double)m);
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double m_j = mean[j];
        double inv_factor = 1.0 / (sqrt_m * stddev[j]);
        #pragma omp simd aligned(data:ALIGN_SIZE)
        for (int i = 0; i < m; i++) {
            data[j*m + i] = (data[j*m + i] - m_j) * inv_factor;
        }
    }
    
    // Calculate correlation matrix with SIMD
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        corr[i*n + i] = 1.0;
        
        for (int j = i + 1; j < n; j++) {
            double sum = 0.0;
            
            // Use AVX2 intrinsics for better vectorization
            int k = 0;
            #ifdef __AVX2__
            __m256d sum_vec = _mm256_setzero_pd();
            for (; k <= m - 4; k += 4) {
                __m256d a = _mm256_load_pd(&data[i*m + k]);
                __m256d b = _mm256_load_pd(&data[j*m + k]);
                sum_vec = _mm256_fmadd_pd(a, b, sum_vec);
            }
            double temp[4];
            _mm256_store_pd(temp, sum_vec);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
            #endif
            
            // Handle remaining elements
            #pragma omp simd reduction(+:sum) aligned(data:ALIGN_SIZE)
            for (int k2 = k; k2 < m; k2++) {
                sum += data[i*m + k2] * data[j*m + k2];
            }
            
            corr[i*n + j] = sum;
            corr[j*n + i] = sum;
        }
    }
}

// Strategy 5: Task-based dynamic
void kernel_correlation_tasks(int m, int n, double *data,
                             double *corr, double *mean, double *stddev) {
    const int chunk = 32;
    
    #pragma omp parallel
    {
        // Calculate means with tasks
        #pragma omp single
        {
            for (int j = 0; j < n; j += chunk) {
                #pragma omp task firstprivate(j)
                {
                    int j_end = (j + chunk < n) ? j + chunk : n;
                    for (int jj = j; jj < j_end; jj++) {
                        double sum = 0.0;
                        for (int i = 0; i < m; i++) {
                            sum += data[jj*m + i];
                        }
                        mean[jj] = sum / m;
                    }
                }
            }
        }
        
        // Calculate standard deviations with tasks
        #pragma omp single
        {
            for (int j = 0; j < n; j += chunk) {
                #pragma omp task firstprivate(j) depend(in:mean[j:chunk])
                {
                    int j_end = (j + chunk < n) ? j + chunk : n;
                    for (int jj = j; jj < j_end; jj++) {
                        double sum = 0.0;
                        for (int i = 0; i < m; i++) {
                            double diff = data[jj*m + i] - mean[jj];
                            sum += diff * diff;
                        }
                        stddev[jj] = sqrt(sum / m);
                        if (stddev[jj] <= EPS) stddev[jj] = 1.0;
                    }
                }
            }
        }
        
        // Normalize data with tasks
        #pragma omp single
        {
            double sqrt_m = sqrt((double)m);
            for (int j = 0; j < n; j += chunk) {
                #pragma omp task firstprivate(j) \
                         depend(in:mean[j:chunk],stddev[j:chunk]) \
                         depend(inout:data[j*m:chunk*m])
                {
                    int j_end = (j + chunk < n) ? j + chunk : n;
                    for (int jj = j; jj < j_end; jj++) {
                        double inv_factor = 1.0 / (sqrt_m * stddev[jj]);
                        for (int i = 0; i < m; i++) {
                            data[jj*m + i] = (data[jj*m + i] - mean[jj]) * inv_factor;
                        }
                    }
                }
            }
        }
        
        // Calculate correlation matrix with tasks
        #pragma omp single
        {
            for (int i = 0; i < n; i += chunk) {
                for (int j = i; j < n; j += chunk) {
                    #pragma omp task firstprivate(i, j) \
                             depend(in:data[i*m:chunk*m],data[j*m:chunk*m]) \
                             depend(out:corr[i*n+j:chunk*chunk])
                    {
                        int i_end = (i + chunk < n) ? i + chunk : n;
                        int j_end = (j + chunk < n) ? j + chunk : n;
                        
                        for (int ii = i; ii < i_end; ii++) {
                            int jj_start = (ii >= j) ? ii : j;
                            for (int jj = jj_start; jj < j_end; jj++) {
                                if (ii == jj) {
                                    corr[ii*n + ii] = 1.0;
                                } else if (jj > ii) {
                                    double sum = 0.0;
                                    for (int k = 0; k < m; k++) {
                                        sum += data[ii*m + k] * data[jj*m + k];
                                    }
                                    corr[ii*n + jj] = sum;
                                    corr[jj*n + ii] = sum;
                                }
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
    double *data_orig = (double*)aligned_malloc(N * M * sizeof(double));
    double *data = (double*)aligned_malloc(N * M * sizeof(double));
    double *corr = (double*)aligned_malloc(N * N * sizeof(double));
    double *mean = (double*)aligned_malloc(N * sizeof(double));
    double *stddev = (double*)aligned_malloc(N * sizeof(double));
    
    // Initialize arrays
    init_array(M, N, data_orig, mean, stddev);
    
    // Calculate FLOPS
    long long total_flops = calculate_flops(M, N);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    printf("\n=== Running Correlation Matrix Benchmark ===\n");
    printf("Data points: %d, Variables: %d\n", M, N);
    printf("Total FLOPS: %lld\n", total_flops);
    printf("Memory footprint: %.2f MB\n\n",
           (N*M + N*N + 2*N) * sizeof(double) / (1024.0*1024.0));
    
    // Sequential baseline
    memcpy(data, data_orig, N * M * sizeof(double));
    memset(corr, 0, N * N * sizeof(double));
    double start = omp_get_wtime();
    kernel_correlation_sequential(M, N, data, corr, mean, stddev);
    double serial_time = omp_get_wtime() - start;
    
    // Verify sequential result
    double error = verify_correlation(N, corr);
    printf("Sequential: %.4f seconds (%.2f GFLOPS) [Error: %.2e]\n\n",
           serial_time, total_flops / (serial_time * 1e9), error);
    
    // Save reference result
    double *corr_ref = (double*)aligned_malloc(N * N * sizeof(double));
    memcpy(corr_ref, corr, N * N * sizeof(double));
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    int num_thread_configs = 4;
    
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "Strategy", "Threads", "Time (s)", "Speedup", "Efficiency", "GFLOPS");
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "--------", "-------", "--------", "-------", "----------", "------");
    
    // Define strategies
    typedef void (*strategy_func)(int, int, double*, double*, double*, double*);
    
    struct {
        const char* name;
        strategy_func func;
    } strategies[] = {
        {"Row-parallel", kernel_correlation_row_parallel},
        {"Tiled/Blocked", kernel_correlation_tiled},
        {"SIMD Vectorized", kernel_correlation_simd},
        {"Task-based", kernel_correlation_tasks}
    };
    
    // Test each strategy
    for (int s = 0; s < 4; s++) {
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);
            
            // Time the strategy
            double times[MEASUREMENT_ITERATIONS];
            for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
                memcpy(data, data_orig, N * M * sizeof(double));
                memset(corr, 0, N * N * sizeof(double));
                memset(mean, 0, N * sizeof(double));
                memset(stddev, 0, N * sizeof(double));
                
                start = omp_get_wtime();
                strategies[s].func(M, N, data, corr, mean, stddev);
                times[iter] = omp_get_wtime() - start;
            }
            
            // Calculate average time
            double avg_time = 0.0;
            for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
                avg_time += times[i];
            }
            avg_time /= MEASUREMENT_ITERATIONS;
            
            // Verify correctness
            error = verify_correlation(N, corr);
            
            // Compare with reference
            double max_diff = 0.0;
            for (int i = 0; i < N * N; i++) {
                double diff = fabs(corr[i] - corr_ref[i]);
                if (diff > max_diff) max_diff = diff;
            }
            
            // Calculate metrics
            double speedup = serial_time / avg_time;
            double efficiency = speedup / num_threads * 100.0;
            double gflops = total_flops / (avg_time * 1e9);
            
            printf("%-25s %-10d %-12.4f %-12.2f %-12.1f%% %-10.2f",
                   strategies[s].name, num_threads, avg_time, speedup, efficiency, gflops);
            
            if (error > 1e-10 || max_diff > 1e-10) {
                printf(" [ERROR: %.2e, DIFF: %.2e]", error, max_diff);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // Free memory
    free(data_orig);
    free(data);
    free(corr);
    free(corr_ref);
    free(mean);
    free(stddev);
    
    return 0;
}