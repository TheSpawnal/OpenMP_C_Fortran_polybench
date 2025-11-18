/**
 * PolyBench/C - 2D Dynamic Programming Benchmark
 * Multiple OpenMP parallelization strategies for sequence alignment
 * 
 * Based on Smith-Waterman-like local alignment algorithm
 * Classic irregular parallel pattern with diagonal dependencies
 * 
 * Compilation: gcc -O3 -march=native -fopenmp -o dynprog dynprog.c -lm
 * Execution: ./dynprog [sequence_length] [num_threads]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Default problem sizes */
#ifndef SEQUENCE_LENGTH
#define SEQUENCE_LENGTH 4000
#endif

/* Memory alignment for SIMD */
#define ALIGN_SIZE 64

/* Scoring parameters */
#define MATCH_SCORE 2
#define MISMATCH_PENALTY -1
#define GAP_PENALTY -1

/* Tiling parameters */
#define TILE_SIZE 64
#define MIN_STRIPE 50

/* Function prototypes */
void initialize_sequences(int n, char *seq1, char *seq2);
void initialize_dp_matrix(int n, int *dp);
int max3(int a, int b, int c);

/* Strategy implementations */
void dynprog_serial(int n, char *seq1, char *seq2, int *dp);
void dynprog_wavefront_basic(int n, char *seq1, char *seq2, int *dp);
void dynprog_wavefront_dynamic(int n, char *seq1, char *seq2, int *dp);
void dynprog_tiled_deps(int n, char *seq1, char *seq2, int *dp, int tile_size);
void dynprog_task_based(int n, char *seq1, char *seq2, int *dp, int min_size);
void dynprog_hybrid(int n, char *seq1, char *seq2, int *dp, int stripe_width);
void dynprog_ordered_sink(int n, char *seq1, char *seq2, int *dp);

/* Verification and utility functions */
double verify_results(int n, int *dp_test, int *dp_ref);
void print_benchmark_header(void);
void run_benchmark(const char *name, int n, char *seq1, char *seq2, 
                   int *dp, int *dp_ref, int num_threads,
                   void (*func)(int, char*, char*, int*));

/**
 * Initialize random DNA sequences (A, C, G, T)
 */
void initialize_sequences(int n, char *seq1, char *seq2) {
    const char bases[] = {'A', 'C', 'G', 'T'};
    unsigned int seed = 42;
    
    #pragma omp parallel
    {
        unsigned int local_seed = seed + omp_get_thread_num();
        
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            seq1[i] = bases[rand_r(&local_seed) % 4];
        }
        
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            seq2[i] = bases[rand_r(&local_seed) % 4];
        }
    }
}

/**
 * Initialize DP matrix with zeros
 */
void initialize_dp_matrix(int n, int *dp) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            dp[i * (n + 1) + j] = 0;
        }
    }
}

/**
 * Return maximum of three integers
 */
inline int max3(int a, int b, int c) {
    int max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}

/**
 * STRATEGY 0: Serial reference implementation
 * Classic Smith-Waterman algorithm with simple scoring
 */
void dynprog_serial(int n, char *seq1, char *seq2, int *dp) {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int match = dp[(i-1) * (n + 1) + (j-1)] + 
                       (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
            int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
            int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
            
            dp[i * (n + 1) + j] = max3(match, delete, insert);
            if (dp[i * (n + 1) + j] < 0) {
                dp[i * (n + 1) + j] = 0;
            }
        }
    }
}

/**
 * STRATEGY 1: Wavefront parallelization with static scheduling
 * Processes anti-diagonals in parallel
 * All elements on same anti-diagonal are independent
 */
void dynprog_wavefront_basic(int n, char *seq1, char *seq2, int *dp) {
    // Process each anti-diagonal
    for (int wave = 2; wave <= 2 * n; wave++) {
        int start_i = (wave > n + 1) ? wave - n : 1;
        int end_i = (wave <= n + 1) ? wave - 1 : n;
        
        #pragma omp parallel for schedule(static)
        for (int i = start_i; i <= end_i; i++) {
            int j = wave - i;
            if (j >= 1 && j <= n) {
                int match = dp[(i-1) * (n + 1) + (j-1)] + 
                           (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
                int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
                int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
                
                dp[i * (n + 1) + j] = max3(match, delete, insert);
                if (dp[i * (n + 1) + j] < 0) {
                    dp[i * (n + 1) + j] = 0;
                }
            }
        }
    }
}

/**
 * STRATEGY 2: Wavefront with dynamic scheduling
 * Better load balancing for irregular workloads
 * Uses dynamic scheduling with small chunks
 */
void dynprog_wavefront_dynamic(int n, char *seq1, char *seq2, int *dp) {
    for (int wave = 2; wave <= 2 * n; wave++) {
        int start_i = (wave > n + 1) ? wave - n : 1;
        int end_i = (wave <= n + 1) ? wave - 1 : n;
        int chunk_size = (end_i - start_i + 1) / (omp_get_max_threads() * 4);
        if (chunk_size < 1) chunk_size = 1;
        
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = start_i; i <= end_i; i++) {
            int j = wave - i;
            if (j >= 1 && j <= n) {
                int match = dp[(i-1) * (n + 1) + (j-1)] + 
                           (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
                int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
                int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
                
                dp[i * (n + 1) + j] = max3(match, delete, insert);
                if (dp[i * (n + 1) + j] < 0) {
                    dp[i * (n + 1) + j] = 0;
                }
            }
        }
    }
}

/**
 * STRATEGY 3: Tiled wavefront with cache blocking
 * Process tiles in wavefront order for better cache locality
 * Balances parallelism with memory hierarchy
 */
void dynprog_tiled_deps(int n, char *seq1, char *seq2, int *dp, int tile_size) {
    int num_tiles = (n + tile_size - 1) / tile_size;
    
    // Process tile wavefronts
    for (int tile_wave = 0; tile_wave < 2 * num_tiles - 1; tile_wave++) {
        int start_ti = (tile_wave >= num_tiles) ? tile_wave - num_tiles + 1 : 0;
        int end_ti = (tile_wave < num_tiles) ? tile_wave : num_tiles - 1;
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (int ti = start_ti; ti <= end_ti; ti++) {
            int tj = tile_wave - ti;
            if (tj >= 0 && tj < num_tiles) {
                int i_start = ti * tile_size + 1;
                int i_end = ((ti + 1) * tile_size < n) ? (ti + 1) * tile_size : n;
                int j_start = tj * tile_size + 1;
                int j_end = ((tj + 1) * tile_size < n) ? (tj + 1) * tile_size : n;
                
                // Process this tile
                for (int i = i_start; i <= i_end; i++) {
                    #pragma omp simd
                    for (int j = j_start; j <= j_end; j++) {
                        int match = dp[(i-1) * (n + 1) + (j-1)] + 
                                   (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
                        int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
                        int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
                        
                        dp[i * (n + 1) + j] = max3(match, delete, insert);
                        if (dp[i * (n + 1) + j] < 0) {
                            dp[i * (n + 1) + j] = 0;
                        }
                    }
                }
            }
        }
    }
}

/**
 * STRATEGY 4: Task-based recursive divide-and-conquer
 * Uses OpenMP tasks with dependency tracking
 * Good for deep parallel hierarchies
 */
void dynprog_task_recursive(int n, char *seq1, char *seq2, int *dp,
                            int i_start, int i_end, int j_start, int j_end,
                            int min_size) {
    int rows = i_end - i_start + 1;
    int cols = j_end - j_start + 1;
    
    if (rows <= min_size || cols <= min_size) {
        // Base case: compute serially
        for (int i = i_start; i <= i_end; i++) {
            for (int j = j_start; j <= j_end; j++) {
                int match = dp[(i-1) * (n + 1) + (j-1)] + 
                           (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
                int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
                int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
                
                dp[i * (n + 1) + j] = max3(match, delete, insert);
                if (dp[i * (n + 1) + j] < 0) {
                    dp[i * (n + 1) + j] = 0;
                }
            }
        }
    } else {
        // Divide into quadrants and process in dependency order
        int i_mid = (i_start + i_end) / 2;
        int j_mid = (j_start + j_end) / 2;
        
        // Top-left (depends on nothing)
        #pragma omp task shared(dp, seq1, seq2) if(rows * cols > min_size * 2)
        dynprog_task_recursive(n, seq1, seq2, dp, i_start, i_mid, j_start, j_mid, min_size);
        
        #pragma omp taskwait
        
        // Top-right and bottom-left (depend on top-left)
        #pragma omp task shared(dp, seq1, seq2) if(rows * cols > min_size * 2)
        dynprog_task_recursive(n, seq1, seq2, dp, i_start, i_mid, j_mid + 1, j_end, min_size);
        
        #pragma omp task shared(dp, seq1, seq2) if(rows * cols > min_size * 2)
        dynprog_task_recursive(n, seq1, seq2, dp, i_mid + 1, i_end, j_start, j_mid, min_size);
        
        #pragma omp taskwait
        
        // Bottom-right (depends on all others)
        #pragma omp task shared(dp, seq1, seq2) if(rows * cols > min_size * 2)
        dynprog_task_recursive(n, seq1, seq2, dp, i_mid + 1, i_end, j_mid + 1, j_end, min_size);
        
        #pragma omp taskwait
    }
}

void dynprog_task_based(int n, char *seq1, char *seq2, int *dp, int min_size) {
    #pragma omp parallel
    {
        #pragma omp single
        dynprog_task_recursive(n, seq1, seq2, dp, 1, n, 1, n, min_size);
    }
}

/**
 * STRATEGY 5: Hybrid strided approach
 * Combines row striding with SIMD vectorization
 * Reduces synchronization overhead
 */
void dynprog_hybrid(int n, char *seq1, char *seq2, int *dp, int stripe_width) {
    int num_stripes = (n + stripe_width - 1) / stripe_width;
    
    // Process in stripe groups
    for (int stripe_group = 0; stripe_group < num_stripes; stripe_group++) {
        int row_start = stripe_group * stripe_width + 1;
        int row_end = ((stripe_group + 1) * stripe_width < n) ? 
                      (stripe_group + 1) * stripe_width : n;
        
        // Within each stripe group, use wavefront
        for (int local_wave = 2; local_wave <= stripe_width + n; local_wave++) {
            #pragma omp parallel for schedule(dynamic, 4)
            for (int local_i = 1; local_i <= row_end - row_start + 1; local_i++) {
                int i = row_start + local_i - 1;
                int j = local_wave - local_i;
                
                if (i >= row_start && i <= row_end && j >= 1 && j <= n) {
                    // Ensure dependencies are satisfied
                    if (stripe_group == 0 || i > row_start || 
                        (stripe_group > 0 && local_wave > stripe_width + 1)) {
                        int match = dp[(i-1) * (n + 1) + (j-1)] + 
                                   (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
                        int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
                        int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
                        
                        dp[i * (n + 1) + j] = max3(match, delete, insert);
                        if (dp[i * (n + 1) + j] < 0) {
                            dp[i * (n + 1) + j] = 0;
                        }
                    }
                }
            }
        }
    }
}

/**
 * STRATEGY 6: OpenMP ordered directive with doacross dependencies
 * Uses OpenMP 4.5+ ordered(2) for explicit dependency tracking
 * Most portable but potentially limited performance
 */
void dynprog_ordered_sink(int n, char *seq1, char *seq2, int *dp) {
    #pragma omp parallel for ordered(2) schedule(dynamic, 16)
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            // Wait for dependencies
            #pragma omp ordered depend(sink: i-1, j) depend(sink: i, j-1)
            
            int match = dp[(i-1) * (n + 1) + (j-1)] + 
                       (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_PENALTY);
            int delete = dp[(i-1) * (n + 1) + j] + GAP_PENALTY;
            int insert = dp[i * (n + 1) + (j-1)] + GAP_PENALTY;
            
            dp[i * (n + 1) + j] = max3(match, delete, insert);
            if (dp[i * (n + 1) + j] < 0) {
                dp[i * (n + 1) + j] = 0;
            }
            
            // Signal completion
            #pragma omp ordered depend(source)
        }
    }
}

/**
 * Verify results against reference implementation
 */
double verify_results(int n, int *dp_test, int *dp_ref) {
    double max_diff = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(max:max_diff)
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            double diff = fabs((double)(dp_test[i * (n + 1) + j] - 
                                       dp_ref[i * (n + 1) + j]));
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    return max_diff;
}

/**
 * Print benchmark header
 */
void print_benchmark_header(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  2D Dynamic Programming - OpenMP Benchmark Suite\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Problem: Smith-Waterman Local Sequence Alignment\n");
    printf("  Dependencies: Each cell depends on top, left, and diagonal\n");
    printf("  Challenge: Wavefront parallelization with load balancing\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
}

/**
 * Run a single benchmark strategy
 */
void run_benchmark(const char *name, int n, char *seq1, char *seq2, 
                   int *dp, int *dp_ref, int num_threads,
                   void (*func)(int, char*, char*, int*)) {
    initialize_dp_matrix(n, dp);
    
    double start_time = omp_get_wtime();
    func(n, seq1, seq2, dp);
    double end_time = omp_get_wtime();
    
    double elapsed = end_time - start_time;
    double cells_per_sec = ((double)n * n) / elapsed / 1e6;
    double diff = verify_results(n, dp, dp_ref);
    
    printf("%-30s %8.4f sec  %8.2f Mcells/s  Score: %6d  Error: %.0e\n",
           name, elapsed, cells_per_sec, dp[n * (n + 1) + n], diff);
}

/**
 * Main benchmark driver
 */
int main(int argc, char *argv[]) {
    int n = SEQUENCE_LENGTH;
    int num_threads = omp_get_max_threads();
    
    // Parse command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    
    omp_set_num_threads(num_threads);
    
    print_benchmark_header();
    printf("Configuration:\n");
    printf("  Sequence length:     %d\n", n);
    printf("  Matrix size:         %d x %d\n", n, n);
    printf("  Total cells:         %ld\n", (long)n * n);
    printf("  Threads:             %d\n", num_threads);
    printf("  Tile size:           %d\n", TILE_SIZE);
    printf("  Min stripe width:    %d\n", MIN_STRIPE);
    printf("\n");
    
    // Allocate aligned memory
    char *seq1 = (char *)aligned_alloc(ALIGN_SIZE, n * sizeof(char));
    char *seq2 = (char *)aligned_alloc(ALIGN_SIZE, n * sizeof(char));
    int *dp = (int *)aligned_alloc(ALIGN_SIZE, (n + 1) * (n + 1) * sizeof(int));
    int *dp_ref = (int *)aligned_alloc(ALIGN_SIZE, (n + 1) * (n + 1) * sizeof(int));
    
    if (!seq1 || !seq2 || !dp || !dp_ref) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize data
    printf("Initializing sequences and computing reference...\n");
    initialize_sequences(n, seq1, seq2);
    initialize_dp_matrix(n, dp_ref);
    
    // Compute reference solution
    double ref_start = omp_get_wtime();
    dynprog_serial(n, seq1, seq2, dp_ref);
    double ref_time = omp_get_wtime() - ref_start;
    int ref_score = dp_ref[n * (n + 1) + n];
    
    printf("Reference (serial) computed in %.4f seconds (score: %d)\n\n", 
           ref_time, ref_score);
    
    printf("%-30s %12s %16s %14s %12s\n", 
           "Strategy", "Time", "Throughput", "Final Score", "Error");
    printf("───────────────────────────────────────────────────────────────\n");
    
    // Warmup
    initialize_dp_matrix(n, dp);
    dynprog_wavefront_basic(n, seq1, seq2, dp);
    
    // Run all strategies
    run_benchmark("Wavefront (static)", n, seq1, seq2, dp, dp_ref, 
                  num_threads, dynprog_wavefront_basic);
    
    run_benchmark("Wavefront (dynamic)", n, seq1, seq2, dp, dp_ref,
                  num_threads, dynprog_wavefront_dynamic);
    
    // Tiled strategy with custom wrapper
    initialize_dp_matrix(n, dp);
    double start = omp_get_wtime();
    dynprog_tiled_deps(n, seq1, seq2, dp, TILE_SIZE);
    double elapsed = omp_get_wtime() - start;
    double cells_per_sec = ((double)n * n) / elapsed / 1e6;
    double diff = verify_results(n, dp, dp_ref);
    printf("%-30s %8.4f sec  %8.2f Mcells/s  Score: %6d  Error: %.0e\n",
           "Tiled wavefront", elapsed, cells_per_sec, 
           dp[n * (n + 1) + n], diff);
    
    // Task-based strategy with custom wrapper
    initialize_dp_matrix(n, dp);
    start = omp_get_wtime();
    dynprog_task_based(n, seq1, seq2, dp, MIN_STRIPE);
    elapsed = omp_get_wtime() - start;
    cells_per_sec = ((double)n * n) / elapsed / 1e6;
    diff = verify_results(n, dp, dp_ref);
    printf("%-30s %8.4f sec  %8.2f Mcells/s  Score: %6d  Error: %.0e\n",
           "Task-based recursive", elapsed, cells_per_sec,
           dp[n * (n + 1) + n], diff);
    
    // Hybrid strategy with custom wrapper
    initialize_dp_matrix(n, dp);
    start = omp_get_wtime();
    dynprog_hybrid(n, seq1, seq2, dp, MIN_STRIPE);
    elapsed = omp_get_wtime() - start;
    cells_per_sec = ((double)n * n) / elapsed / 1e6;
    diff = verify_results(n, dp, dp_ref);
    printf("%-30s %8.4f sec  %8.2f Mcells/s  Score: %6d  Error: %.0e\n",
           "Hybrid striped", elapsed, cells_per_sec,
           dp[n * (n + 1) + n], diff);
    
    // Ordered directive strategy (may be slow)
    if (n <= 2000) {  // Only run for smaller problems
        run_benchmark("Ordered depend (doacross)", n, seq1, seq2, dp, dp_ref,
                      num_threads, dynprog_ordered_sink);
    }
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("\nKey Observations:\n");
    printf("  • Wavefront methods expose diagonal parallelism\n");
    printf("  • Tiling improves cache locality but adds complexity\n");
    printf("  • Task-based approach good for recursive divide-and-conquer\n");
    printf("  • Dynamic scheduling helps with load imbalance\n");
    printf("  • Ordered directives ensure correctness but limit speedup\n\n");
    
    // Cleanup
    free(seq1);
    free(seq2);
    free(dp);
    free(dp_ref);
    
    return 0;
}
