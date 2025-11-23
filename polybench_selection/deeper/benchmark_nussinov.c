/**
 * nussinov.c: Nussinov Dynamic Programming Benchmark with Multiple Strategies  
 * RNA secondary structure prediction using dynamic programming
 * 
 * Strategies implemented:
 * 1. Sequential (baseline)
 * 2. Wavefront/Anti-diagonal parallel
 * 3. Tiled wavefront
 * 4. Task-based with dependencies
 * 5. Pipeline parallel
 * 6. SIMD wavefront
 * 7. Hybrid coarse+fine grained
 * 8. Cache-optimized blocked
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
#define N 60
#elif defined(SMALL)
#define N 180
#elif defined(MEDIUM)
#define N 500
#elif defined(LARGE)
#define N 2500
#else // Default STANDARD
#define N 1000
#endif

#define ALIGN_SIZE 64
#define TILE_SIZE 32
#define MIN_TASK_SIZE 50

// Base type for RNA sequence
typedef enum { A = 0, C = 1, G = 2, U = 3 } base;

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

// Initialize RNA sequence
static void init_array(int n, base *seq, int *table) {
    // Initialize random RNA sequence
    for (int i = 0; i < n; i++) {
        seq[i] = (base)((i + 1) % 4);
    }
    
    // Initialize DP table
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            table[i*n + j] = 0;
        }
    }
}

// Check if two bases can pair (Watson-Crick pairing)
static inline int can_pair(base b1, base b2) {
    // A-U and C-G pairs
    return ((b1 == A && b2 == U) || (b1 == U && b2 == A) ||
            (b1 == C && b2 == G) || (b1 == G && b2 == C)) ? 1 : 0;
}

// Maximum of two integers
static inline int max(int a, int b) {
    return (a > b) ? a : b;
}

// Verify result by comparing with reference
static double verify_result(int n, int *table_ref, int *table_test) {
    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int error = abs(table_ref[i*n + j] - table_test[i*n + j]);
            if (error > max_error) max_error = error;
        }
    }
    return max_error;
}

// Calculate FLOPS for Nussinov
static long long calculate_flops(int n) {
    // For each cell (i,j): 
    // - 3 max operations for cases 2,3,4
    // - j-i iterations for bifurcation (case 4), each with max and add
    // Total: approximately n^3 operations
    return (long long)n * n * n;
}

// Strategy 1: Sequential baseline (reference implementation)
void kernel_nussinov_sequential(int n, base *seq, int *table) {
    // Fill the DP table
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            // Case 1: i and j pair
            if (j - 1 >= 0 && i + 1 < n) {
                if (i < j - 1) {
                    table[i*n + j] = max(table[i*n + j], 
                                       table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
                } else {
                    table[i*n + j] = max(table[i*n + j], can_pair(seq[i], seq[j]));
                }
            }
            
            // Case 2: i unpaired
            if (i + 1 < n) {
                table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
            }
            
            // Case 3: j unpaired
            if (j - 1 >= 0) {
                table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
            }
            
            // Case 4: bifurcation at k
            for (int k = i + 1; k < j; k++) {
                table[i*n + j] = max(table[i*n + j], 
                                   table[i*n + k] + table[(k+1)*n + j]);
            }
        }
    }
}

// Strategy 2: Wavefront/Anti-diagonal parallel
void kernel_nussinov_wavefront(int n, base *seq, int *table) {
    // Process anti-diagonals
    for (int diag = 1; diag < n; diag++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n - diag; i++) {
            int j = i + diag;
            
            // Case 1: i and j pair
            if (diag > 1) {
                table[i*n + j] = max(table[i*n + j],
                                   table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
            } else if (diag == 1) {
                table[i*n + j] = can_pair(seq[i], seq[j]);
            }
            
            // Case 2: i unpaired
            if (i + 1 < n) {
                table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
            }
            
            // Case 3: j unpaired  
            if (j - 1 >= i) {
                table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
            }
            
            // Case 4: bifurcation at k
            int local_max = table[i*n + j];
            #pragma omp simd reduction(max:local_max)
            for (int k = i + 1; k < j; k++) {
                local_max = max(local_max, table[i*n + k] + table[(k+1)*n + j]);
            }
            table[i*n + j] = local_max;
        }
    }
}

// Strategy 3: Tiled wavefront
void kernel_nussinov_tiled_wavefront(int n, base *seq, int *table) {
    const int tile = TILE_SIZE;
    
    // Process by tile diagonals
    for (int tile_diag = 0; tile_diag < (2 * n / tile); tile_diag++) {
        #pragma omp parallel for schedule(dynamic)
        for (int tile_i = 0; tile_i <= tile_diag; tile_i++) {
            int tile_j = tile_diag - tile_i;
            
            int i_start = tile_i * tile;
            int j_start = tile_j * tile;
            
            if (i_start >= n || j_start >= n) continue;
            
            int i_end = (i_start + tile < n) ? i_start + tile : n;
            int j_end = (j_start + tile < n) ? j_start + tile : n;
            
            // Process tile in wavefront order
            for (int diag = 1; diag < tile * 2; diag++) {
                for (int i = i_start; i < i_end; i++) {
                    int j = j_start + (diag - (i - i_start));
                    
                    if (j < j_start || j >= j_end || j <= i || j >= n) continue;
                    
                    // Case 1: i and j pair
                    if (j - i > 1) {
                        table[i*n + j] = max(table[i*n + j],
                                           table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
                    } else if (j - i == 1) {
                        table[i*n + j] = can_pair(seq[i], seq[j]);
                    }
                    
                    // Case 2: i unpaired
                    if (i + 1 < n) {
                        table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
                    }
                    
                    // Case 3: j unpaired
                    if (j - 1 >= i) {
                        table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
                    }
                    
                    // Case 4: bifurcation
                    for (int k = i + 1; k < j; k++) {
                        table[i*n + j] = max(table[i*n + j],
                                           table[i*n + k] + table[(k+1)*n + j]);
                    }
                }
            }
        }
    }
}

// Strategy 4: Task-based with dependencies
void kernel_nussinov_tasks(int n, base *seq, int *table) {
    #pragma omp parallel
    #pragma omp single
    {
        // Process anti-diagonals with tasks
        for (int diag = 1; diag < n; diag++) {
            int num_elements = n - diag;
            int chunk = (num_elements > MIN_TASK_SIZE) ? MIN_TASK_SIZE : num_elements;
            
            for (int start = 0; start < num_elements; start += chunk) {
                int end = (start + chunk < num_elements) ? start + chunk : num_elements;
                
                #pragma omp task firstprivate(diag, start, end) \
                         depend(in: table[(start+1)*n+(start+diag-1):chunk*n], \
                                   table[start*n+(start+diag-1):chunk*n]) \
                         depend(out: table[start*n+(start+diag):chunk])
                {
                    for (int i = start; i < end; i++) {
                        int j = i + diag;
                        
                        // Case 1: i and j pair
                        if (diag > 1) {
                            table[i*n + j] = max(table[i*n + j],
                                               table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
                        } else if (diag == 1) {
                            table[i*n + j] = can_pair(seq[i], seq[j]);
                        }
                        
                        // Case 2: i unpaired
                        if (i + 1 < n) {
                            table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
                        }
                        
                        // Case 3: j unpaired
                        if (j - 1 >= i) {
                            table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
                        }
                        
                        // Case 4: bifurcation
                        for (int k = i + 1; k < j; k++) {
                            table[i*n + j] = max(table[i*n + j],
                                               table[i*n + k] + table[(k+1)*n + j]);
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
    }
}

// Strategy 5: Pipeline parallel
void kernel_nussinov_pipeline(int n, base *seq, int *table) {
    int num_threads = omp_get_max_threads();
    int stripe_width = n / (num_threads * 2);
    if (stripe_width < 32) stripe_width = 32;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // Each thread processes stripes in pipeline fashion
        for (int stage = 0; stage < n * 2; stage++) {
            for (int stripe = tid; stripe * stripe_width < n; stripe += num_threads) {
                int i_start = stripe * stripe_width;
                int i_end = (i_start + stripe_width < n) ? i_start + stripe_width : n;
                
                // Process current stage of stripe
                for (int i = i_start; i < i_end; i++) {
                    int j = i + (stage - stripe);
                    
                    if (j <= i || j >= n) continue;
                    
                    // Case 1: i and j pair
                    if (j - i > 1) {
                        table[i*n + j] = max(table[i*n + j],
                                           table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
                    } else if (j - i == 1) {
                        table[i*n + j] = can_pair(seq[i], seq[j]);
                    }
                    
                    // Case 2: i unpaired
                    if (i + 1 < n) {
                        table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
                    }
                    
                    // Case 3: j unpaired
                    if (j - 1 >= i) {
                        table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
                    }
                    
                    // Case 4: bifurcation
                    int local_max = table[i*n + j];
                    for (int k = i + 1; k < j; k++) {
                        local_max = max(local_max, table[i*n + k] + table[(k+1)*n + j]);
                    }
                    table[i*n + j] = local_max;
                }
            }
            
            // Synchronize after each stage
            #pragma omp barrier
        }
    }
}

// Strategy 6: SIMD wavefront
void kernel_nussinov_simd_wavefront(int n, base *seq, int *table) {
    // Process anti-diagonals with SIMD
    for (int diag = 1; diag < n; diag++) {
        int num_elements = n - diag;
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_elements; i++) {
            int j = i + diag;
            
            // Cases 1-3 (same as before)
            if (diag > 1) {
                table[i*n + j] = max(table[i*n + j],
                                   table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
            } else if (diag == 1) {
                table[i*n + j] = can_pair(seq[i], seq[j]);
            }
            
            if (i + 1 < n) {
                table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
            }
            
            if (j - 1 >= i) {
                table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
            }
            
            // Case 4: bifurcation with SIMD
            int local_max = table[i*n + j];
            
            // Process with SIMD where possible
            int k;
            #ifdef __AVX2__
            __m256i max_vec = _mm256_set1_epi32(local_max);
            for (k = i + 1; k <= j - 4; k += 4) {
                __m256i left = _mm256_loadu_si256((__m256i*)&table[i*n + k]);
                __m256i right = _mm256_loadu_si256((__m256i*)&table[(k+1)*n + j]);
                right = _mm256_shuffle_epi32(right, 0x1B); // Reverse for alignment
                __m256i sum = _mm256_add_epi32(left, right);
                max_vec = _mm256_max_epi32(max_vec, sum);
            }
            
            // Extract maximum
            int temp[8];
            _mm256_storeu_si256((__m256i*)temp, max_vec);
            for (int t = 0; t < 4; t++) {
                local_max = max(local_max, temp[t]);
            }
            #else
            k = i + 1;
            #endif
            
            // Handle remaining elements
            for (; k < j; k++) {
                local_max = max(local_max, table[i*n + k] + table[(k+1)*n + j]);
            }
            
            table[i*n + j] = local_max;
        }
    }
}

// Strategy 7: Hybrid coarse+fine grained
void kernel_nussinov_hybrid(int n, base *seq, int *table) {
    const int coarse_tile = 128;
    const int fine_tile = 32;
    
    // Coarse-grained tiling
    for (int c_diag = 0; c_diag < (2 * n / coarse_tile); c_diag++) {
        #pragma omp parallel for schedule(dynamic)
        for (int c_i = 0; c_i <= c_diag; c_i++) {
            int c_j = c_diag - c_i;
            
            int ci_start = c_i * coarse_tile;
            int cj_start = c_j * coarse_tile;
            
            if (ci_start >= n || cj_start >= n) continue;
            
            int ci_end = (ci_start + coarse_tile < n) ? ci_start + coarse_tile : n;
            int cj_end = (cj_start + coarse_tile < n) ? cj_start + coarse_tile : n;
            
            // Fine-grained processing within coarse tile
            for (int f_diag = 0; f_diag < (2 * coarse_tile / fine_tile); f_diag++) {
                #pragma omp parallel for schedule(static) if(ci_end - ci_start > 64)
                for (int f_i = 0; f_i <= f_diag; f_i++) {
                    int f_j = f_diag - f_i;
                    
                    int fi_start = ci_start + f_i * fine_tile;
                    int fj_start = cj_start + f_j * fine_tile;
                    
                    if (fi_start >= ci_end || fj_start >= cj_end) continue;
                    
                    int fi_end = (fi_start + fine_tile < ci_end) ? fi_start + fine_tile : ci_end;
                    int fj_end = (fj_start + fine_tile < cj_end) ? fj_start + fine_tile : cj_end;
                    
                    // Process fine tile
                    for (int i = fi_start; i < fi_end; i++) {
                        for (int j = fj_start; j < fj_end; j++) {
                            if (j <= i) continue;
                            
                            // All four cases
                            if (j - i > 1) {
                                table[i*n + j] = max(table[i*n + j],
                                                   table[(i+1)*n + (j-1)] + can_pair(seq[i], seq[j]));
                            } else if (j - i == 1) {
                                table[i*n + j] = can_pair(seq[i], seq[j]);
                            }
                            
                            if (i + 1 < n) {
                                table[i*n + j] = max(table[i*n + j], table[(i+1)*n + j]);
                            }
                            
                            if (j - 1 >= i) {
                                table[i*n + j] = max(table[i*n + j], table[i*n + (j-1)]);
                            }
                            
                            // Bifurcation with cache blocking
                            int local_max = table[i*n + j];
                            for (int kb = i + 1; kb < j; kb += fine_tile) {
                                int k_end = (kb + fine_tile < j) ? kb + fine_tile : j;
                                #pragma omp simd reduction(max:local_max)
                                for (int k = kb; k < k_end; k++) {
                                    local_max = max(local_max, table[i*n + k] + table[(k+1)*n + j]);
                                }
                            }
                            table[i*n + j] = local_max;
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
    base *seq = (base*)aligned_malloc(N * sizeof(base));
    int *table_ref = (int*)aligned_malloc(N * N * sizeof(int));
    int *table = (int*)aligned_malloc(N * N * sizeof(int));
    
    // Initialize sequence and table
    init_array(N, seq, table_ref);
    
    // Calculate FLOPS
    long long total_flops = calculate_flops(N);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    printf("\n=== Running Nussinov Dynamic Programming Benchmark ===\n");
    printf("Sequence length: %d\n", N);
    printf("Total FLOPS: %lld\n", total_flops);
    printf("Memory footprint: %.2f MB\n\n",
           (N * sizeof(base) + 2 * N * N * sizeof(int)) / (1024.0*1024.0));
    
    // Sequential baseline
    double start = omp_get_wtime();
    kernel_nussinov_sequential(N, seq, table_ref);
    double serial_time = omp_get_wtime() - start;
    printf("Sequential: %.4f seconds (%.2f GFLOPS)\n\n",
           serial_time, total_flops / (serial_time * 1e9));
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    int num_thread_configs = 4;
    
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "Strategy", "Threads", "Time (s)", "Speedup", "Efficiency", "GFLOPS");
    printf("%-25s %-10s %-12s %-12s %-12s %-10s\n",
           "--------", "-------", "--------", "-------", "----------", "------");
    
    // Define strategies
    typedef void (*strategy_func)(int, base*, int*);
    
    struct {
        const char* name;
        strategy_func func;
    } strategies[] = {
        {"Wavefront", kernel_nussinov_wavefront},
        {"Tiled Wavefront", kernel_nussinov_tiled_wavefront},
        {"Task-based", kernel_nussinov_tasks},
        {"Pipeline", kernel_nussinov_pipeline},
        {"SIMD Wavefront", kernel_nussinov_simd_wavefront},
        {"Hybrid", kernel_nussinov_hybrid}
    };
    
    // Test each strategy
    for (int s = 0; s < 6; s++) {
        for (int t = 0; t < num_thread_configs; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);
            
            // Time the strategy
            double times[MEASUREMENT_ITERATIONS];
            for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
                memset(table, 0, N * N * sizeof(int));
                start = omp_get_wtime();
                strategies[s].func(N, seq, table);
                times[iter] = omp_get_wtime() - start;
            }
            
            // Calculate average time
            double avg_time = 0.0;
            for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
                avg_time += times[i];
            }
            avg_time /= MEASUREMENT_ITERATIONS;
            
            // Verify correctness
            memset(table, 0, N * N * sizeof(int));
            strategies[s].func(N, seq, table);
            double error = verify_result(N, table_ref, table);
            
            // Calculate metrics
            double speedup = serial_time / avg_time;
            double efficiency = speedup / num_threads * 100.0;
            double gflops = total_flops / (avg_time * 1e9);
            
            printf("%-25s %-10d %-12.4f %-12.2f %-12.1f%% %-10.2f",
                   strategies[s].name, num_threads, avg_time, speedup, efficiency, gflops);
            
            if (error > 0) {
                printf(" [ERROR: %.0f]", error);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // Print final score
    printf("Maximum alignment score: %d\n", table_ref[0*N + (N-1)]);
    
    // Free memory
    free(seq);
    free(table_ref);
    free(table);
    
    return 0;
}