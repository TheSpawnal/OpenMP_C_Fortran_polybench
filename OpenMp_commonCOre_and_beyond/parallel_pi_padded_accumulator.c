#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000000;  // 100M steps for meaningful timing
double step;

// PAD 8 means 8 doubles = 64 bytes (typical cache line size)
#define PAD 8
#define MAX_THREADS 16

// Version 1: WITHOUT padding (suffers from false sharing)
void calculate_pi_no_padding(int num_threads)
{      
    int i, nthreads; 
    double pi;
    double *sum = (double*) calloc(num_threads, sizeof(double));
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;
        
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + nthrds){
            x = (i + 0.5)*step;
            sum[id] += 4.0/(1.0 + x*x);
        }
    }
    
    end_time = omp_get_wtime();
    
    for (i = 0, pi = 0.0; i < nthreads; i++) 
        pi += sum[i] * step;
    
    printf("NO PAD  - Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
    
    free(sum);
}

// Version 2: WITH padding (eliminates false sharing)
void calculate_pi_with_padding(int num_threads)
{      
    int i, nthreads; 
    double pi;
    double sum[MAX_THREADS][PAD];  // Each thread gets its own cache line
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;
        
        for (i = id, sum[id][0] = 0.0; i < num_steps; i = i + nthrds){
            x = (i + 0.5)*step;
            sum[id][0] += 4.0/(1.0 + x*x);  // Only use first element of padded array
        }
    }
    
    end_time = omp_get_wtime();
    
    for (i = 0, pi = 0.0; i < nthreads; i++) 
        pi += sum[i][0] * step;  // Access first element
    
    printf("PADDED  - Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
}

int main()
{
    printf("=============================================================\n");
    printf("FALSE SHARING DEMONSTRATION - Pi Calculation\n");
    printf("=============================================================\n");
    printf("Problem size: %ld steps\n", num_steps);
    printf("Cache line size: 64 bytes (%d doubles)\n", PAD);
    printf("=============================================================\n\n");
    
    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = 4;
    
    for (int t = 0; t < num_tests; t++) {
        int threads = thread_counts[t];
        
        printf("--- Testing with %d thread(s) ---\n", threads);
        calculate_pi_no_padding(threads);
        calculate_pi_with_padding(threads);
        printf("\n");
    }
    
    printf("=============================================================\n");
    printf("EXPLANATION:\n");
    printf("- WITHOUT padding: sum[0], sum[1], etc. are adjacent in memory\n");
    printf("  They share cache lines, causing 'false sharing'\n");
    printf("- WITH padding: Each sum[i][0] is 64 bytes apart\n");
    printf("  Each thread gets its own cache line, no false sharing!\n");
    printf("=============================================================\n");
    
    return 0;
}