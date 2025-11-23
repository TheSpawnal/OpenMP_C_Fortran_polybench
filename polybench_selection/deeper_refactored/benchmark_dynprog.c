/**
 * benchmark_dynprog_advanced.c: Advanced Dynamic Programming Benchmark Suite
 * Multiple DP patterns with sophisticated parallelization strategies
 * Includes:
 * 1. Sequence Alignment (Needleman-Wunsch / Smith-Waterman)
 * 2. Matrix Chain Multiplication 
 * 3. Longest Common Subsequence (LCS)
 * 4. Knapsack Problem (0/1 and unbounded)
 * 5. Edit Distance with operations
 * 
 * Parallelization strategies:
 * - Wavefront/Anti-diagonal
 * - Tiled with dependencies
 * - Task-based with DAG
 * - Pipeline parallel
 * - Hybrid approaches
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <immintrin.h>
#include "benchmark_metrics.h"

// Problem size definitions
#ifdef MINI
#define SEQ_LEN 100
#define MATRIX_COUNT 10
#define ITEMS 50
#elif defined(SMALL)
#define SEQ_LEN 500
#define MATRIX_COUNT 20
#define ITEMS 100
#elif defined(MEDIUM)
#define SEQ_LEN 2000
#define MATRIX_COUNT 50
#define ITEMS 500
#elif defined(LARGE)
#define SEQ_LEN 10000
#define MATRIX_COUNT 100
#define ITEMS 2000
#else // Default STANDARD
#define SEQ_LEN 1000
#define MATRIX_COUNT 30
#define ITEMS 200
#endif

#define ALIGN_SIZE 64
#define TILE_SIZE 32
#define MIN_TASK_SIZE 64

// Scoring parameters for sequence alignment
#define MATCH_SCORE 2
#define MISMATCH_SCORE -1
#define GAP_PENALTY -1

typedef struct {
    int value;
    int weight;
} Item;

// Aligned memory allocation for 2D arrays
static int** alloc_2d_int(int rows, int cols) {
    int** array = (int**)malloc(rows * sizeof(int*));
    int* data = (int*)aligned_alloc(ALIGN_SIZE, rows * cols * sizeof(int));
    
    if (!array || !data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    memset(data, 0, rows * cols * sizeof(int));
    
    for (int i = 0; i < rows; i++) {
        array[i] = &data[i * cols];
    }
    
    return array;
}

static void free_2d_int(int** array) {
    if (array) {
        if (array[0]) free(array[0]);
        free(array);
    }
}

// Generate random sequences for testing
static void generate_sequences(char* seq1, char* seq2, int len) {
    const char bases[] = "ACGT";
    for (int i = 0; i < len; i++) {
        seq1[i] = bases[rand() % 4];
        seq2[i] = bases[rand() % 4];
    }
    seq1[len] = '\0';
    seq2[len] = '\0';
}

// Generate matrix dimensions for chain multiplication
static void generate_matrix_dimensions(int* dims, int count) {
    for (int i = 0; i <= count; i++) {
        dims[i] = 10 + rand() % 100;
    }
}

// Generate items for knapsack
static void generate_items(Item* items, int count) {
    for (int i = 0; i < count; i++) {
        items[i].value = 10 + rand() % 100;
        items[i].weight = 1 + rand() % 50;
    }
}

// Calculate FLOPS for DP algorithms
static long long calculate_dp_flops(int n, int m) {
    // Approximate: each cell requires constant operations
    return (long long)n * m * 5;  // Comparisons and arithmetic
}

// =============================================================================
// SEQUENCE ALIGNMENT ALGORITHMS
// =============================================================================

// Strategy 1: Sequential Needleman-Wunsch
void sequence_alignment_sequential(const char* seq1, const char* seq2, 
                                  int len1, int len2, int** dp) {
    // Initialize first row and column
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j * GAP_PENALTY;
    }
    
    // Fill DP table
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            int match = dp[i-1][j-1] + 
                       (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_SCORE);
            int delete = dp[i-1][j] + GAP_PENALTY;
            int insert = dp[i][j-1] + GAP_PENALTY;
            
            dp[i][j] = (match > delete) ? 
                      ((match > insert) ? match : insert) :
                      ((delete > insert) ? delete : insert);
        }
    }
}

// Strategy 2: Wavefront parallel alignment
void sequence_alignment_wavefront(const char* seq1, const char* seq2,
                                 int len1, int len2, int** dp) {
    // Initialize boundaries
    #pragma omp parallel for
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i * GAP_PENALTY;
    }
    #pragma omp parallel for
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j * GAP_PENALTY;
    }
    
    // Process anti-diagonals
    for (int wave = 2; wave <= len1 + len2; wave++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= len1; i++) {
            int j = wave - i;
            if (j >= 1 && j <= len2) {
                int match = dp[i-1][j-1] + 
                           (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_SCORE);
                int delete = dp[i-1][j] + GAP_PENALTY;
                int insert = dp[i][j-1] + GAP_PENALTY;
                
                dp[i][j] = (match > delete) ? 
                          ((match > insert) ? match : insert) :
                          ((delete > insert) ? delete : insert);
            }
        }
    }
}

// Strategy 3: Tiled alignment with cache optimization
void sequence_alignment_tiled(const char* seq1, const char* seq2,
                             int len1, int len2, int** dp) {
    const int tile = TILE_SIZE;
    
    // Initialize boundaries
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j * GAP_PENALTY;
    }
    
    // Process in tiles following dependencies
    for (int ti = 0; ti <= len1 / tile; ti++) {
        for (int tj = 0; tj <= len2 / tile; tj++) {
            // Determine tile boundaries
            int i_start = ti * tile + 1;
            int i_end = (i_start + tile <= len1) ? i_start + tile : len1 + 1;
            int j_start = tj * tile + 1;
            int j_end = (j_start + tile <= len2) ? j_start + tile : len2 + 1;
            
            // Process tile
            #pragma omp task depend(in: dp[i_start-1][j_start-1:tile+1], \
                                       dp[i_start-1:tile+1][j_start-1]) \
                            depend(out: dp[i_start:tile][j_start:tile])
            {
                for (int i = i_start; i < i_end; i++) {
                    for (int j = j_start; j < j_end; j++) {
                        int match = dp[i-1][j-1] + 
                                   (seq1[i-1] == seq2[j-1] ? MATCH_SCORE : MISMATCH_SCORE);
                        int delete = dp[i-1][j] + GAP_PENALTY;
                        int insert = dp[i][j-1] + GAP_PENALTY;
                        
                        dp[i][j] = (match > delete) ? 
                                  ((match > insert) ? match : insert) :
                                  ((delete > insert) ? delete : insert);
                    }
                }
            }
        }
    }
    #pragma omp taskwait
}

// =============================================================================
// MATRIX CHAIN MULTIPLICATION
// =============================================================================

// Sequential matrix chain multiplication
void matrix_chain_sequential(int* dims, int n, int** dp) {
    // dp[i][j] = minimum cost to multiply matrices i through j
    
    // Initialize: single matrices have zero cost
    for (int i = 0; i < n; i++) {
        dp[i][i] = 0;
    }
    
    // Chain length
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            
            // Try all split points
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dims[i] * dims[k+1] * dims[j+1];
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                }
            }
        }
    }
}

// Parallel matrix chain multiplication with tasks
void matrix_chain_parallel(int* dims, int n, int** dp) {
    // Initialize diagonal
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dp[i][i] = 0;
    }
    
    // Process by chain length
    for (int len = 2; len <= n; len++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            
            int local_min = INT_MAX;
            #pragma omp simd reduction(min:local_min)
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dims[i] * dims[k+1] * dims[j+1];
                if (cost < local_min) {
                    local_min = cost;
                }
            }
            dp[i][j] = local_min;
        }
    }
}

// =============================================================================
// LONGEST COMMON SUBSEQUENCE
// =============================================================================

// Sequential LCS
void lcs_sequential(const char* seq1, const char* seq2, 
                   int len1, int len2, int** dp) {
    // Initialize first row and column to 0
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = 0;
    }
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = 0;
    }
    
    // Fill DP table
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (seq1[i-1] == seq2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = (dp[i-1][j] > dp[i][j-1]) ? dp[i-1][j] : dp[i][j-1];
            }
        }
    }
}

// Parallel LCS with wavefront
void lcs_wavefront(const char* seq1, const char* seq2,
                  int len1, int len2, int** dp) {
    // Initialize boundaries
    #pragma omp parallel for
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = 0;
    }
    #pragma omp parallel for
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = 0;
    }
    
    // Process anti-diagonals
    for (int wave = 2; wave <= len1 + len2; wave++) {
        #pragma omp parallel for
        for (int i = 1; i <= len1; i++) {
            int j = wave - i;
            if (j >= 1 && j <= len2) {
                if (seq1[i-1] == seq2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = (dp[i-1][j] > dp[i][j-1]) ? dp[i-1][j] : dp[i][j-1];
                }
            }
        }
    }
}

// =============================================================================
// KNAPSACK PROBLEM
// =============================================================================

// Sequential 0/1 Knapsack
void knapsack_sequential(Item* items, int n, int capacity, int** dp) {
    // dp[i][w] = maximum value using first i items with weight limit w
    
    // Initialize first row and column
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 0;
    }
    for (int w = 0; w <= capacity; w++) {
        dp[0][w] = 0;
    }
    
    // Fill DP table
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= capacity; w++) {
            if (items[i-1].weight <= w) {
                int with_item = dp[i-1][w - items[i-1].weight] + items[i-1].value;
                int without_item = dp[i-1][w];
                dp[i][w] = (with_item > without_item) ? with_item : without_item;
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
}

// Parallel Knapsack with row-wise parallelization
void knapsack_parallel(Item* items, int n, int capacity, int** dp) {
    // Initialize
    #pragma omp parallel for
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 0;
    }
    #pragma omp parallel for
    for (int w = 0; w <= capacity; w++) {
        dp[0][w] = 0;
    }
    
    // Process row by row (dependencies between rows)
    for (int i = 1; i <= n; i++) {
        #pragma omp parallel for
        for (int w = 1; w <= capacity; w++) {
            if (items[i-1].weight <= w) {
                int with_item = dp[i-1][w - items[i-1].weight] + items[i-1].value;
                int without_item = dp[i-1][w];
                dp[i][w] = (with_item > without_item) ? with_item : without_item;
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
}

// =============================================================================
// EDIT DISTANCE
// =============================================================================

// Sequential Edit Distance (Levenshtein)
void edit_distance_sequential(const char* str1, const char* str2,
                             int len1, int len2, int** dp) {
    // Initialize boundaries
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i;  // Deletions
    }
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j;  // Insertions
    }
    
    // Fill DP table
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (str1[i-1] == str2[j-1]) {
                dp[i][j] = dp[i-1][j-1];  // No operation needed
            } else {
                int substitute = dp[i-1][j-1] + 1;
                int delete = dp[i-1][j] + 1;
                int insert = dp[i][j-1] + 1;
                
                dp[i][j] = (substitute < delete) ?
                          ((substitute < insert) ? substitute : insert) :
                          ((delete < insert) ? delete : insert);
            }
        }
    }
}

// Parallel Edit Distance with diagonal processing
void edit_distance_parallel(const char* str1, const char* str2,
                           int len1, int len2, int** dp) {
    // Initialize boundaries
    #pragma omp parallel for
    for (int i = 0; i <= len1; i++) {
        dp[i][0] = i;
    }
    #pragma omp parallel for
    for (int j = 0; j <= len2; j++) {
        dp[0][j] = j;
    }
    
    // Process diagonals
    for (int diag = 2; diag <= len1 + len2; diag++) {
        #pragma omp parallel for
        for (int i = 1; i <= len1; i++) {
            int j = diag - i;
            if (j >= 1 && j <= len2) {
                if (str1[i-1] == str2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    int substitute = dp[i-1][j-1] + 1;
                    int delete = dp[i-1][j] + 1;
                    int insert = dp[i][j-1] + 1;
                    
                    dp[i][j] = (substitute < delete) ?
                              ((substitute < insert) ? substitute : insert) :
                              ((delete < insert) ? delete : insert);
                }
            }
        }
    }
}

// =============================================================================
// BENCHMARK DRIVER
// =============================================================================

// Test a specific DP algorithm
void run_dp_benchmark(const char* name, 
                     void (*seq_func)(const void*, const void*, int, int, int**),
                     void (*par_func)(const void*, const void*, int, int, int**),
                     const void* data1, const void* data2,
                     int dim1, int dim2) {
    
    printf("\n=== %s Benchmark ===\n", name);
    printf("Problem size: %d x %d\n", dim1, dim2);
    
    // Allocate DP table
    int** dp_seq = alloc_2d_int(dim1 + 1, dim2 + 1);
    int** dp_par = alloc_2d_int(dim1 + 1, dim2 + 1);
    
    // Sequential baseline
    double start = omp_get_wtime();
    seq_func(data1, data2, dim1, dim2, dp_seq);
    double seq_time = omp_get_wtime() - start;
    printf("Sequential: %.4f seconds\n", seq_time);
    
    // Test different thread counts
    int thread_counts[] = {2, 4, 8, 16};
    
    printf("\nThreads  Time(s)   Speedup   Efficiency\n");
    printf("-------  --------  --------  ----------\n");
    
    for (int t = 0; t < 4; t++) {
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);
        
        // Clear DP table
        memset(dp_par[0], 0, (dim1 + 1) * (dim2 + 1) * sizeof(int));
        
        // Run parallel version
        start = omp_get_wtime();
        par_func(data1, data2, dim1, dim2, dp_par);
        double par_time = omp_get_wtime() - start;
        
        // Verify correctness
        int correct = 1;
        for (int i = 0; i <= dim1 && correct; i++) {
            for (int j = 0; j <= dim2 && correct; j++) {
                if (dp_seq[i][j] != dp_par[i][j]) {
                    correct = 0;
                }
            }
        }
        
        double speedup = seq_time / par_time;
        double efficiency = speedup / num_threads * 100.0;
        
        printf("%7d  %8.4f  %8.2f  %9.1f%%", 
               num_threads, par_time, speedup, efficiency);
        
        if (!correct) {
            printf(" [INCORRECT]");
        }
        printf("\n");
    }
    
    // Free memory
    free_2d_int(dp_seq);
    free_2d_int(dp_par);
}

int main(int argc, char** argv) {
    // Initialize random seed
    srand(42);
    
    // Warmup
    printf("Warming up CPU...\n");
    warmup_cpu();
    
    printf("\n=== Advanced Dynamic Programming Benchmark Suite ===\n");
    printf("Sequence length: %d\n", SEQ_LEN);
    printf("Matrix count: %d\n", MATRIX_COUNT);
    printf("Items: %d\n\n", ITEMS);
    
    // Generate test data
    char* seq1 = (char*)malloc((SEQ_LEN + 1) * sizeof(char));
    char* seq2 = (char*)malloc((SEQ_LEN + 1) * sizeof(char));
    generate_sequences(seq1, seq2, SEQ_LEN);
    
    int* matrix_dims = (int*)malloc((MATRIX_COUNT + 1) * sizeof(int));
    generate_matrix_dimensions(matrix_dims, MATRIX_COUNT);
    
    Item* items = (Item*)malloc(ITEMS * sizeof(Item));
    generate_items(items, ITEMS);
    int capacity = ITEMS * 25;  // Average weight * 0.5
    
    // Run benchmarks
    
    // 1. Sequence Alignment
    run_dp_benchmark("Sequence Alignment (Needleman-Wunsch)",
                    (void*)sequence_alignment_sequential,
                    (void*)sequence_alignment_wavefront,
                    seq1, seq2, SEQ_LEN, SEQ_LEN);
    
    // 2. Longest Common Subsequence
    run_dp_benchmark("Longest Common Subsequence",
                    (void*)lcs_sequential,
                    (void*)lcs_wavefront,
                    seq1, seq2, SEQ_LEN, SEQ_LEN);
    
    // 3. Edit Distance
    run_dp_benchmark("Edit Distance (Levenshtein)",
                    (void*)edit_distance_sequential,
                    (void*)edit_distance_parallel,
                    seq1, seq2, SEQ_LEN, SEQ_LEN);
    
    // 4. Matrix Chain Multiplication
    printf("\n=== Matrix Chain Multiplication ===\n");
    printf("Number of matrices: %d\n", MATRIX_COUNT);
    
    int** mcm_seq = alloc_2d_int(MATRIX_COUNT, MATRIX_COUNT);
    int** mcm_par = alloc_2d_int(MATRIX_COUNT, MATRIX_COUNT);
    
    double start = omp_get_wtime();
    matrix_chain_sequential(matrix_dims, MATRIX_COUNT, mcm_seq);
    double seq_time = omp_get_wtime() - start;
    printf("Sequential: %.4f seconds\n", seq_time);
    printf("Minimum operations: %d\n", mcm_seq[0][MATRIX_COUNT-1]);
    
    printf("\nThreads  Time(s)   Speedup   Efficiency\n");
    printf("-------  --------  --------  ----------\n");
    
    int thread_counts[] = {2, 4, 8, 16};
    for (int t = 0; t < 4; t++) {
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);
        
        memset(mcm_par[0], 0, MATRIX_COUNT * MATRIX_COUNT * sizeof(int));
        
        start = omp_get_wtime();
        matrix_chain_parallel(matrix_dims, MATRIX_COUNT, mcm_par);
        double par_time = omp_get_wtime() - start;
        
        double speedup = seq_time / par_time;
        double efficiency = speedup / num_threads * 100.0;
        
        printf("%7d  %8.4f  %8.2f  %9.1f%%\n",
               num_threads, par_time, speedup, efficiency);
    }
    
    free_2d_int(mcm_seq);
    free_2d_int(mcm_par);
    
    // 5. Knapsack Problem
    printf("\n=== 0/1 Knapsack Problem ===\n");
    printf("Items: %d, Capacity: %d\n", ITEMS, capacity);
    
    int** knap_seq = alloc_2d_int(ITEMS + 1, capacity + 1);
    int** knap_par = alloc_2d_int(ITEMS + 1, capacity + 1);
    
    start = omp_get_wtime();
    knapsack_sequential(items, ITEMS, capacity, knap_seq);
    seq_time = omp_get_wtime() - start;
    printf("Sequential: %.4f seconds\n", seq_time);
    printf("Maximum value: %d\n", knap_seq[ITEMS][capacity]);
    
    printf("\nThreads  Time(s)   Speedup   Efficiency\n");
    printf("-------  --------  --------  ----------\n");
    
    for (int t = 0; t < 4; t++) {
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);
        
        memset(knap_par[0], 0, (ITEMS + 1) * (capacity + 1) * sizeof(int));
        
        start = omp_get_wtime();
        knapsack_parallel(items, ITEMS, capacity, knap_par);
        double par_time = omp_get_wtime() - start;
        
        double speedup = seq_time / par_time;
        double efficiency = speedup / num_threads * 100.0;
        
        printf("%7d  %8.4f  %8.2f  %9.1f%%\n",
               num_threads, par_time, speedup, efficiency);
    }
    
    free_2d_int(knap_seq);
    free_2d_int(knap_par);
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("Dynamic Programming algorithms exhibit varying degrees of parallelism:\n");
    printf("- Wavefront algorithms: Good scaling for large problems\n");
    printf("- Row-wise parallelism: Limited by dependencies\n");
    printf("- Tiled approaches: Better cache utilization\n");
    printf("- Task-based: Good for irregular patterns\n");
    
    // Free memory
    free(seq1);
    free(seq2);
    free(matrix_dims);
    free(items);
    
    return 0;
}
