#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

#define PAD 8
#define MAX_THREADS 16

//V1: Original SPMD with strided access
void calculate_pi_strided(int num_threads){      
    int i, nthreads; 
    double pi;
    double *sum = (double*) calloc(num_threads, sizeof(double));
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel{
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;
        
        // STRIDED: Thread 0 does 0,4,8,12... Thread 1 does 1,5,9,13...
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + nthrds){
            x = (i + 0.5)*step;
            sum[id] += 4.0/(1.0 + x*x);
        }
    }
    
    end_time = omp_get_wtime();
    
    for (i = 0, pi = 0.0; i < nthreads; i++) 
        pi += sum[i] * step;
    
    printf("STRIDED - Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
    
    free(sum);
}

// v2: Block distribution (better cache locality)
void calculate_pi_blocked(int num_threads){      
    int i, nthreads; 
    double pi;
    double *sum = (double*) calloc(num_threads, sizeof(double));
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel{
        int i, id, nthrds;
        double x;
        long start, end;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;
        
        // BLOCKED: Divide work into contiguous chunks
        long chunk_size = num_steps / nthrds;
        start = id * chunk_size;
        end = (id == nthrds - 1) ? num_steps : start + chunk_size;
        
        sum[id] = 0.0;
        for (i = start; i < end; i++){
            x = (i + 0.5)*step;
            sum[id] += 4.0/(1.0 + x*x);
        }
    }
    
    end_time = omp_get_wtime();
    
    for (i = 0, pi = 0.0; i < nthreads; i++) 
        pi += sum[i] * step;
    
    printf("BLOCKED - Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
    
    free(sum);
}

// V3: Use OpenMP reduction
void calculate_pi_reduction(int num_threads)
{      
    double pi = 0.0;
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        double x, local_sum = 0.0;
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        
        // BLOCKED distribution with private local_sum
        long chunk_size = num_steps / nthrds;
        long start = id * chunk_size;
        long end = (id == nthrds - 1) ? num_steps : start + chunk_size;
        
        for (long i = start; i < end; i++){
            x = (i + 0.5)*step;
            local_sum += 4.0/(1.0 + x*x);
        }
        
        // Safe reduction at the end (no false sharing during computation)
        #pragma omp atomic
        pi += local_sum * step;
    }
    
    end_time = omp_get_wtime();
    
    printf("REDUCE  - Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
}

// V4: OpenMP for loop
void calculate_pi_parallel_for(int num_threads){      
    double pi = 0.0;
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        double sum = 0.0;
        
        #pragma omp for
        for (long i = 0; i < num_steps; i++){
            double x = (i + 0.5)*step;
            sum += 4.0/(1.0 + x*x);
        }
        
        #pragma omp atomic
        pi += sum;
    }
    
    pi *= step;
    end_time = omp_get_wtime();
    
    printf("PAR-FOR - Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
}

int main()
{
    printf("=============================================================\n");
    printf("PERFORMANCE COMPARISON - Different Parallel Strategies\n");
    printf("=============================================================\n");
    printf("Problem size: %ld steps\n", num_steps);
    printf("CPU: Intel i5-10210U (4 cores, 8 threads)\n");
    printf("=============================================================\n\n");
    
    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = 4;
    
    for (int t = 0; t < num_tests; t++) {
        int threads = thread_counts[t];
        
        printf("--- Testing with %d thread(s) ---\n", threads);
        calculate_pi_strided(threads);
        calculate_pi_blocked(threads);
        calculate_pi_reduction(threads);
        calculate_pi_parallel_for(threads);
        printf("\n");
    }
    
    printf("=============================================================\n");
    printf("EXPLANATION:\n");
    printf("STRIDED:  Poor cache locality (threads jump around memory)\n");
    printf("BLOCKED:  Good cache locality (each thread gets contiguous chunk)\n");
    printf("REDUCE:   Uses private variables + atomic reduction\n");
    printf("PAR-FOR:  OpenMP handles everything automatically (BEST!)\n");
    printf("=============================================================\n");
    
    return 0;
}