/*
 * Correlation Matrix Computation Benchmark
 * Multiple parallelization strategies for data mining workloads
 * Computes Pearson correlation coefficients between all pairs of variables
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>

#define EPS 0.1

// Strategy 1: Basic parallel with row-wise computation
void correlation_rowwise(int m, int n, double *data, double *corr, 
                        double *mean, double *stddev) {
    // Compute means
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += data[i * n + j];
        }
        mean[j] = sum / m;
    }
    
    // Compute standard deviations
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            double diff = data[i * n + j] - mean[j];
            sum += diff * diff;
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Compute correlation matrix (upper triangle)
    #pragma omp parallel for schedule(dynamic)
    for (int j1 = 0; j1 < n - 1; j1++) {
        for (int j2 = j1 + 1; j2 < n; j2++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += (data[i * n + j1] - mean[j1]) * 
                       (data[i * n + j2] - mean[j2]);
            }
            corr[j1 * n + j2] = sum / (m * stddev[j1] * stddev[j2]);
            corr[j2 * n + j1] = corr[j1 * n + j2]; // Symmetric
        }
        corr[j1 * n + j1] = 1.0; // Diagonal
    }
    corr[(n-1) * n + (n-1)] = 1.0;
}

// Strategy 2: Tiled computation with cache optimization
void correlation_tiled(int m, int n, double *data, double *corr,
                      double *mean, double *stddev, int tile_size) {
    // Compute means with tiling
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int ii = 0; ii < m; ii += tile_size) {
            int i_end = (ii + tile_size < m) ? ii + tile_size : m;
            for (int i = ii; i < i_end; i++) {
                sum += data[i * n + j];
            }
        }
        mean[j] = sum / m;
    }
    
    // Compute standard deviations with tiling
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int ii = 0; ii < m; ii += tile_size) {
            int i_end = (ii + tile_size < m) ? ii + tile_size : m;
            for (int i = ii; i < i_end; i++) {
                double diff = data[i * n + j] - mean[j];
                sum += diff * diff;
            }
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Compute correlation with 2D tiling
    #pragma omp parallel for schedule(dynamic)
    for (int j1_tile = 0; j1_tile < n; j1_tile += tile_size) {
        for (int j2_tile = j1_tile; j2_tile < n; j2_tile += tile_size) {
            int j1_end = (j1_tile + tile_size < n) ? j1_tile + tile_size : n;
            int j2_end = (j2_tile + tile_size < n) ? j2_tile + tile_size : n;
            
            for (int j1 = j1_tile; j1 < j1_end; j1++) {
                int j2_start = (j1_tile == j2_tile) ? j1 : j2_tile;
                for (int j2 = j2_start; j2 < j2_end; j2++) {
                    if (j1 == j2) {
                        corr[j1 * n + j1] = 1.0;
                    } else {
                        double sum = 0.0;
                        for (int i = 0; i < m; i++) {
                            sum += (data[i * n + j1] - mean[j1]) * 
                                   (data[i * n + j2] - mean[j2]);
                        }
                        corr[j1 * n + j2] = sum / (m * stddev[j1] * stddev[j2]);
                        corr[j2 * n + j1] = corr[j1 * n + j2];
                    }
                }
            }
        }
    }
}

// Strategy 3: SIMD vectorized computation
void correlation_simd(int m, int n, double *data, double *corr,
                     double *mean, double *stddev) {
    // Compute means with SIMD
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            sum += data[i * n + j];
        }
        mean[j] = sum / m;
    }
    
    // Compute standard deviations with SIMD
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        double m_j = mean[j];
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            double diff = data[i * n + j] - m_j;
            sum += diff * diff;
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize data for correlation computation
    double *norm_data = (double *)aligned_alloc(64, m * n * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            norm_data[i * n + j] = (data[i * n + j] - mean[j]) / stddev[j];
        }
    }
    
    // Compute correlation with SIMD
    #pragma omp parallel for schedule(dynamic)
    for (int j1 = 0; j1 < n; j1++) {
        for (int j2 = j1; j2 < n; j2++) {
            if (j1 == j2) {
                corr[j1 * n + j1] = 1.0;
            } else {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int i = 0; i < m; i++) {
                    sum += norm_data[i * n + j1] * norm_data[i * n + j2];
                }
                corr[j1 * n + j2] = sum / m;
                corr[j2 * n + j1] = corr[j1 * n + j2];
            }
        }
    }
    
    free(norm_data);
}

// Strategy 4: Task-based computation with dependency management
void correlation_tasks(int m, int n, double *data, double *corr,
                      double *mean, double *stddev, int chunk_size) {
    // Compute means with tasks
    #pragma omp parallel
    #pragma omp single
    {
        for (int j = 0; j < n; j += chunk_size) {
            #pragma omp task firstprivate(j)
            {
                int j_end = (j + chunk_size < n) ? j + chunk_size : n;
                for (int jj = j; jj < j_end; jj++) {
                    double sum = 0.0;
                    for (int i = 0; i < m; i++) {
                        sum += data[i * n + jj];
                    }
                    mean[jj] = sum / m;
                }
            }
        }
        #pragma omp taskwait
        
        // Compute standard deviations with tasks
        for (int j = 0; j < n; j += chunk_size) {
            #pragma omp task firstprivate(j)
            {
                int j_end = (j + chunk_size < n) ? j + chunk_size : n;
                for (int jj = j; jj < j_end; jj++) {
                    double sum = 0.0;
                    for (int i = 0; i < m; i++) {
                        double diff = data[i * n + jj] - mean[jj];
                        sum += diff * diff;
                    }
                    stddev[jj] = sqrt(sum / m);
                    if (stddev[jj] <= EPS) stddev[jj] = 1.0;
                }
            }
        }
        #pragma omp taskwait
        
        // Compute correlation with tasks
        for (int j1 = 0; j1 < n; j1++) {
            #pragma omp task firstprivate(j1) depend(out:corr[j1*n:n])
            {
                for (int j2 = j1; j2 < n; j2++) {
                    if (j1 == j2) {
                        corr[j1 * n + j1] = 1.0;
                    } else {
                        double sum = 0.0;
                        for (int i = 0; i < m; i++) {
                            sum += (data[i * n + j1] - mean[j1]) * 
                                   (data[i * n + j2] - mean[j2]);
                        }
                        corr[j1 * n + j2] = sum / (m * stddev[j1] * stddev[j2]);
                        corr[j2 * n + j1] = corr[j1 * n + j2];
                    }
                }
            }
        }
    }
}

// Strategy 5: Reduction-based with custom reduction operations
void correlation_reduction(int m, int n, double *data, double *corr,
                          double *mean, double *stddev) {
    // Parallel computation of means and stddevs together
    #pragma omp parallel
    {
        double *local_mean = (double *)calloc(n, sizeof(double));
        double *local_m2 = (double *)calloc(n, sizeof(double));
        
        #pragma omp for nowait
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                local_mean[j] += data[i * n + j];
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < n; j++) {
                mean[j] += local_mean[j];
            }
        }
        
        #pragma omp barrier
        #pragma omp single
        {
            for (int j = 0; j < n; j++) {
                mean[j] /= m;
            }
        }
        
        #pragma omp for nowait
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double diff = data[i * n + j] - mean[j];
                local_m2[j] += diff * diff;
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < n; j++) {
                stddev[j] += local_m2[j];
            }
        }
        
        free(local_mean);
        free(local_m2);
    }
    
    // Finalize standard deviations
    for (int j = 0; j < n; j++) {
        stddev[j] = sqrt(stddev[j] / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Parallel correlation computation
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j1 = 0; j1 < n; j1++) {
        for (int j2 = 0; j2 < n; j2++) {
            if (j1 <= j2) {
                if (j1 == j2) {
                    corr[j1 * n + j1] = 1.0;
                } else {
                    double sum = 0.0;
                    for (int i = 0; i < m; i++) {
                        sum += (data[i * n + j1] - mean[j1]) * 
                               (data[i * n + j2] - mean[j2]);
                    }
                    corr[j1 * n + j2] = sum / (m * stddev[j1] * stddev[j2]);
                    corr[j2 * n + j1] = corr[j1 * n + j2];
                }
            }
        }
    }
}

// Strategy 6: Column-major optimized for better memory access
void correlation_column_major(int m, int n, double *data, double *corr,
                             double *mean, double *stddev) {
    // Transpose data for better cache locality
    double *data_t = (double *)malloc(n * m * sizeof(double));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data_t[j * m + i] = data[i * n + j];
        }
    }
    
    // Compute means (now with better memory access)
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        double *col = &data_t[j * m];
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            sum += col[i];
        }
        mean[j] = sum / m;
    }
    
    // Compute standard deviations
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        double m_j = mean[j];
        double *col = &data_t[j * m];
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            double diff = col[i] - m_j;
            sum += diff * diff;
        }
        stddev[j] = sqrt(sum / m);
        if (stddev[j] <= EPS) stddev[j] = 1.0;
    }
    
    // Normalize columns in place
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double m_j = mean[j];
        double s_j = stddev[j];
        double *col = &data_t[j * m];
        #pragma omp simd
        for (int i = 0; i < m; i++) {
            col[i] = (col[i] - m_j) / s_j;
        }
    }
    
    // Compute correlation (dot products of normalized columns)
    #pragma omp parallel for schedule(dynamic)
    for (int j1 = 0; j1 < n; j1++) {
        double *col1 = &data_t[j1 * m];
        for (int j2 = j1; j2 < n; j2++) {
            if (j1 == j2) {
                corr[j1 * n + j1] = 1.0;
            } else {
                double *col2 = &data_t[j2 * m];
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int i = 0; i < m; i++) {
                    sum += col1[i] * col2[i];
                }
                corr[j1 * n + j2] = sum / m;
                corr[j2 * n + j1] = corr[j1 * n + j2];
            }
        }
    }
    
    free(data_t);
}

// Generate synthetic data
void generate_data(int m, int n, double *data) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data[i * n + j] = ((double)rand() / RAND_MAX) * 100.0;
        }
    }
}

// Verify correlation matrix properties
double verify_correlation(int n, double *corr) {
    double max_error = 0.0;
    
    // Check diagonal elements
    for (int i = 0; i < n; i++) {
        double error = fabs(corr[i * n + i] - 1.0);
        if (error > max_error) max_error = error;
    }
    
    // Check symmetry and bounds
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double error = fabs(corr[i * n + j] - corr[j * n + i]);
            if (error > max_error) max_error = error;
            
            if (fabs(corr[i * n + j]) > 1.0 + 1e-10) {
                max_error = fmax(max_error, fabs(corr[i * n + j]) - 1.0);
            }
        }
    }
    
    return max_error;
}

int main(int argc, char **argv) {
    int m = 2000;  // Number of data points
    int n = 500;   // Number of variables
    
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    
    printf("Correlation Matrix Benchmark: m=%d (data points), n=%d (variables)\n", m, n);
    printf("Threads: %d\n", omp_get_max_threads());
    
    // Allocate memory
    double *data = (double *)malloc(m * n * sizeof(double));
    double *corr = (double *)malloc(n * n * sizeof(double));
    double *mean = (double *)calloc(n, sizeof(double));
    double *stddev = (double *)calloc(n, sizeof(double));
    
    // Generate synthetic data
    generate_data(m, n, data);
    
    double start_time, end_time, error;
    
    // Strategy 1: Row-wise
    memset(mean, 0, n * sizeof(double));
    memset(stddev, 0, n * sizeof(double));
    start_time = omp_get_wtime();
    correlation_rowwise(m, n, data, corr, mean, stddev);
    end_time = omp_get_wtime();
    error = verify_correlation(n, corr);
    printf("Row-wise: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 2: Tiled
    memset(mean, 0, n * sizeof(double));
    memset(stddev, 0, n * sizeof(double));
    start_time = omp_get_wtime();
    correlation_tiled(m, n, data, corr, mean, stddev, 64);
    end_time = omp_get_wtime();
    error = verify_correlation(n, corr);
    printf("Tiled (64): %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 3: SIMD
    memset(mean, 0, n * sizeof(double));
    memset(stddev, 0, n * sizeof(double));
    start_time = omp_get_wtime();
    correlation_simd(m, n, data, corr, mean, stddev);
    end_time = omp_get_wtime();
    error = verify_correlation(n, corr);
    printf("SIMD Vectorized: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 4: Tasks
    memset(mean, 0, n * sizeof(double));
    memset(stddev, 0, n * sizeof(double));
    start_time = omp_get_wtime();
    correlation_tasks(m, n, data, corr, mean, stddev, 50);
    end_time = omp_get_wtime();
    error = verify_correlation(n, corr);
    printf("Task-based: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 5: Reduction
    memset(mean, 0, n * sizeof(double));
    memset(stddev, 0, n * sizeof(double));
    start_time = omp_get_wtime();
    correlation_reduction(m, n, data, corr, mean, stddev);
    end_time = omp_get_wtime();
    error = verify_correlation(n, corr);
    printf("Reduction-based: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    // Strategy 6: Column-major
    memset(mean, 0, n * sizeof(double));
    memset(stddev, 0, n * sizeof(double));
    start_time = omp_get_wtime();
    correlation_column_major(m, n, data, corr, mean, stddev);
    end_time = omp_get_wtime();
    error = verify_correlation(n, corr);
    printf("Column-major: %.4f seconds (error: %.2e)\n", 
           end_time - start_time, error);
    
    free(data);
    free(corr);
    free(mean);
    free(stddev);
    
    return 0;
}
