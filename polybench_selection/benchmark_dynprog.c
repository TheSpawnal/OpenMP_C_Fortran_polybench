/*
 * 2D Dynamic Programming Benchmark (Needleman-Wunsch style)
 * Multiple parallelization strategies for wavefront computation patterns
 * Complex data dependencies with challenging load balancing
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#define MATCH_SCORE 2
#define MISMATCH_SCORE -1
#define GAP_PENALTY -1
#define MIN_VAL -999999

// Strategy 1: Diagonal wavefront parallelization
void dynprog_wavefront(int m, int n, int *seq1, int *seq2, int *dp) {
    // Initialize first row and column
    for (int i = 0; i <= m; i++) {
        dp[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= n; j++) {
        dp[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    // Process diagonals (wavefront)
    int num_diags = m + n - 1;
    for (int d = 1; d <= num_diags; d++) {
        int i_start = (d <= n) ? 1 : d - n + 1;
        int i_end = (d <= m) ? d : m;
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = i_start; i <= i_end; i++) {
            int j = d - i + 1;
            if (j >= 1 && j <= n) {
                int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                
                int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                
                dp[i * (n+1) + j] = fmax(diag, fmax(left, up));
            }
        }
        #pragma omp barrier
    }
}

// Strategy 2: Anti-diagonal blocking with task parallelism
void dynprog_antidiag_tasks(int m, int n, int *seq1, int *seq2, int *dp, int block_size) {
    // Initialize boundaries
    for (int i = 0; i <= m; i++) {
        dp[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= n; j++) {
        dp[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    int num_block_rows = (m + block_size - 1) / block_size;
    int num_block_cols = (n + block_size - 1) / block_size;
    
    #pragma omp parallel
    #pragma omp single
    {
        // Process block anti-diagonals
        for (int d = 0; d < num_block_rows + num_block_cols - 1; d++) {
            int block_i_start = (d < num_block_cols) ? 0 : d - num_block_cols + 1;
            int block_i_end = (d < num_block_rows) ? d : num_block_rows - 1;
            
            for (int block_i = block_i_start; block_i <= block_i_end; block_i++) {
                int block_j = d - block_i;
                if (block_j >= 0 && block_j < num_block_cols) {
                    #pragma omp task firstprivate(block_i, block_j)
                    {
                        int i_start = block_i * block_size + 1;
                        int i_end = fmin((block_i + 1) * block_size, m);
                        int j_start = block_j * block_size + 1;
                        int j_end = fmin((block_j + 1) * block_size, n);
                        
                        for (int i = i_start; i <= i_end; i++) {
                            for (int j = j_start; j <= j_end; j++) {
                                int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                                
                                int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                                int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                                int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                                
                                dp[i * (n+1) + j] = fmax(diag, fmax(left, up));
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
    }
}

// Strategy 3: Row-wise computation with pipeline parallelism
void dynprog_pipeline(int m, int n, int *seq1, int *seq2, int *dp) {
    // Initialize boundaries
    for (int i = 0; i <= m; i++) {
        dp[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= n; j++) {
        dp[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    volatile int *row_done = (volatile int *)calloc(m + 1, sizeof(int));
    row_done[0] = n; // First row is already done
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        for (int i = 1; i <= m; i++) {
            if (i % nthreads == tid) {
                // Wait for previous row to complete enough columns
                for (int j = 1; j <= n; j++) {
                    while (row_done[i-1] < j) {
                        #pragma omp flush
                    }
                    
                    int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                    
                    int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                    int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                    int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                    
                    dp[i * (n+1) + j] = fmax(diag, fmax(left, up));
                    
                    #pragma omp flush
                    row_done[i] = j;
                }
            }
        }
    }
    
    free((void *)row_done);
}

// Strategy 4: Tiled computation with dependency tracking
void dynprog_tiled_deps(int m, int n, int *seq1, int *seq2, int *dp, int tile_size) {
    // Initialize boundaries
    for (int i = 0; i <= m; i++) {
        dp[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= n; j++) {
        dp[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    int num_tiles_i = (m + tile_size - 1) / tile_size;
    int num_tiles_j = (n + tile_size - 1) / tile_size;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int ti = 0; ti < num_tiles_i; ti++) {
                for (int tj = 0; tj < num_tiles_j; tj++) {
                    #pragma omp task depend(in: dp[(ti>0?(ti-1)*tile_size:0)*(n+1)+(tj*tile_size):tile_size], \
                                                dp[(ti*tile_size)*(n+1)+(tj>0?(tj-1)*tile_size:0):tile_size]) \
                                     depend(out: dp[(ti*tile_size)*(n+1)+(tj*tile_size):tile_size*tile_size])
                    {
                        int i_start = ti * tile_size + 1;
                        int i_end = fmin((ti + 1) * tile_size, m);
                        int j_start = tj * tile_size + 1;
                        int j_end = fmin((tj + 1) * tile_size, n);
                        
                        for (int i = i_start; i <= i_end; i++) {
                            for (int j = j_start; j <= j_end; j++) {
                                int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                                
                                int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                                int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                                int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                                
                                dp[i * (n+1) + j] = fmax(diag, fmax(left, up));
                            }
                        }
                    }
                }
            }
        }
    }
}

// Strategy 5: Hybrid approach with coarse-grained wavefront and fine-grained parallelism
void dynprog_hybrid(int m, int n, int *seq1, int *seq2, int *dp, int stripe_width) {
    // Initialize boundaries
    for (int i = 0; i <= m; i++) {
        dp[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= n; j++) {
        dp[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    // Process in stripes (coarse-grained wavefront)
    for (int stripe = 0; stripe < m; stripe += stripe_width) {
        int stripe_end = fmin(stripe + stripe_width, m);
        
        // Within each stripe, use fine-grained parallelism
        for (int i = stripe + 1; i <= stripe_end; i++) {
            #pragma omp parallel for schedule(static) if(n > 100)
            for (int j = 1; j <= n; j++) {
                // Ensure dependencies are met
                if (i == stripe + 1 || j == 1) {
                    // Sequential for stripe boundaries
                    int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                    
                    int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                    int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                    int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                    
                    dp[i * (n+1) + j] = fmax(diag, fmax(left, up));
                } else {
                    // Can be parallel within stripe
                    int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                    
                    int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                    int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                    int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                    
                    dp[i * (n+1) + j] = fmax(diag, fmax(left, up));
                }
            }
        }
    }
}

// Strategy 6: SIMD-optimized wavefront with manual vectorization hints
void dynprog_simd_wavefront(int m, int n, int *seq1, int *seq2, int *dp) {
    // Initialize boundaries
    #pragma omp parallel for simd
    for (int i = 0; i <= m; i++) {
        dp[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    #pragma omp parallel for simd
    for (int j = 0; j <= n; j++) {
        dp[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    // Process with SIMD-friendly access patterns
    for (int d = 2; d <= m + n; d++) {
        int j_start = fmax(1, d - m);
        int j_end = fmin(n, d - 1);
        
        #pragma omp parallel if(j_end - j_start > 32)
        {
            // Process diagonal in chunks suitable for SIMD
            #pragma omp for simd schedule(static)
            for (int j = j_start; j <= j_end; j++) {
                int i = d - j;
                if (i >= 1 && i <= m) {
                    int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
                    
                    int diag = dp[(i-1) * (n+1) + (j-1)] + match;
                    int left = dp[i * (n+1) + (j-1)] + GAP_PENALTY;
                    int up = dp[(i-1) * (n+1) + j] + GAP_PENALTY;
                    
                    int max_val = diag;
                    if (left > max_val) max_val = left;
                    if (up > max_val) max_val = up;
                    
                    dp[i * (n+1) + j] = max_val;
                }
            }
        }
    }
}

// Generate random sequences
void generate_sequences(int m, int n, int *seq1, int *seq2) {
    for (int i = 0; i < m; i++) {
        seq1[i] = rand() % 4; // DNA-like: 0=A, 1=C, 2=G, 3=T
    }
    for (int j = 0; j < n; j++) {
        seq2[j] = rand() % 4;
    }
}

// Verify DP table correctness (sequential reference)
void dynprog_sequential(int m, int n, int *seq1, int *seq2, int *dp_ref) {
    for (int i = 0; i <= m; i++) {
        dp_ref[i * (n + 1) + 0] = i * GAP_PENALTY;
    }
    for (int j = 0; j <= n; j++) {
        dp_ref[0 * (n + 1) + j] = j * GAP_PENALTY;
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int match = (seq1[i-1] == seq2[j-1]) ? MATCH_SCORE : MISMATCH_SCORE;
            
            int diag = dp_ref[(i-1) * (n+1) + (j-1)] + match;
            int left = dp_ref[i * (n+1) + (j-1)] + GAP_PENALTY;
            int up = dp_ref[(i-1) * (n+1) + j] + GAP_PENALTY;
            
            dp_ref[i * (n+1) + j] = fmax(diag, fmax(left, up));
        }
    }
}

// Compare two DP tables
double verify_dp(int m, int n, int *dp, int *dp_ref) {
    double max_diff = 0.0;
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            double diff = abs(dp[i * (n+1) + j] - dp_ref[i * (n+1) + j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    return max_diff;
}

int main(int argc, char **argv) {
    int m = 2000;  // First sequence length
    int n = 2000;  // Second sequence length
    
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    
    printf("2D Dynamic Programming Benchmark: m=%d, n=%d\n", m, n);
    printf("Threads: %d\n", omp_get_max_threads());
    
    // Allocate memory
    int *seq1 = (int *)malloc(m * sizeof(int));
    int *seq2 = (int *)malloc(n * sizeof(int));
    int *dp = (int *)malloc((m + 1) * (n + 1) * sizeof(int));
    int *dp_ref = (int *)malloc((m + 1) * (n + 1) * sizeof(int));
    
    // Generate sequences
    generate_sequences(m, n, seq1, seq2);
    
    // Compute reference solution
    dynprog_sequential(m, n, seq1, seq2, dp_ref);
    int ref_score = dp_ref[m * (n + 1) + n];
    printf("Reference score: %d\n\n", ref_score);
    
    double start_time, end_time;
    double diff;
    
    // Strategy 1: Wavefront
    memset(dp, 0, (m + 1) * (n + 1) * sizeof(int));
    start_time = omp_get_wtime();
    dynprog_wavefront(m, n, seq1, seq2, dp);
    end_time = omp_get_wtime();
    diff = verify_dp(m, n, dp, dp_ref);
    printf("Wavefront: %.4f seconds (score: %d, diff: %.0f)\n", 
           end_time - start_time, dp[m * (n + 1) + n], diff);
    
    // Strategy 2: Anti-diagonal tasks
    memset(dp, 0, (m + 1) * (n + 1) * sizeof(int));
    start_time = omp_get_wtime();
    dynprog_antidiag_tasks(m, n, seq1, seq2, dp, 100);
    end_time = omp_get_wtime();
    diff = verify_dp(m, n, dp, dp_ref);
    printf("Anti-diagonal Tasks (100): %.4f seconds (score: %d, diff: %.0f)\n", 
           end_time - start_time, dp[m * (n + 1) + n], diff);
    
    // Strategy 3: Pipeline
    memset(dp, 0, (m + 1) * (n + 1) * sizeof(int));
    start_time = omp_get_wtime();
    dynprog_pipeline(m, n, seq1, seq2, dp);
    end_time = omp_get_wtime();
    diff = verify_dp(m, n, dp, dp_ref);
    printf("Pipeline: %.4f seconds (score: %d, diff: %.0f)\n", 
           end_time - start_time, dp[m * (n + 1) + n], diff);
    
    // Strategy 4: Tiled with dependencies
    memset(dp, 0, (m + 1) * (n + 1) * sizeof(int));
    start_time = omp_get_wtime();
    dynprog_tiled_deps(m, n, seq1, seq2, dp, 64);
    end_time = omp_get_wtime();
    diff = verify_dp(m, n, dp, dp_ref);
    printf("Tiled Dependencies (64): %.4f seconds (score: %d, diff: %.0f)\n", 
           end_time - start_time, dp[m * (n + 1) + n], diff);
    
    // Strategy 5: Hybrid
    memset(dp, 0, (m + 1) * (n + 1) * sizeof(int));
    start_time = omp_get_wtime();
    dynprog_hybrid(m, n, seq1, seq2, dp, 100);
    end_time = omp_get_wtime();
    diff = verify_dp(m, n, dp, dp_ref);
    printf("Hybrid (stripe=100): %.4f seconds (score: %d, diff: %.0f)\n", 
           end_time - start_time, dp[m * (n + 1) + n], diff);
    
    // Strategy 6: SIMD wavefront
    memset(dp, 0, (m + 1) * (n + 1) * sizeof(int));
    start_time = omp_get_wtime();
    dynprog_simd_wavefront(m, n, seq1, seq2, dp);
    end_time = omp_get_wtime();
    diff = verify_dp(m, n, dp, dp_ref);
    printf("SIMD Wavefront: %.4f seconds (score: %d, diff: %.0f)\n", 
           end_time - start_time, dp[m * (n + 1) + n], diff);
    
    free(seq1);
    free(seq2);
    free(dp);
    free(dp_ref);
    
    return 0;
}
