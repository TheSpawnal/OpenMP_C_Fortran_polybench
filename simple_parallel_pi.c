// #include <stdio.h>
// #include <omp.h>
// static long num_steps = 100000; 
// double step;
// #define NUM_THREADS 2
// void main()
// {      int i, nthreads; double pi, sum[NUM_THREADS];
//         step = 1.0/(double) num_steps;
//         omp_set_num_threads(NUM_THREADS);
//     #pragma omp parallel
//     {
//         int i, id , nthrds;
//         double x;
//         id = omp_get_thread_num();
//         nthrds = omp_get_num_threads();
//         if (id == 0) nthreads = nthrds;
//         for (i = id , sum[id] = 0.0 ; i< num_steps; i = i + nthrds){
//             x = (i + 0.5)*step;
//             sum[id] += 4.0/(1.0 + x*x);
//         }
//     }
//         for (i = 0, pi = 0.0 ; i< nthreads; i++) pi += sum[i] * step;
//         printf("Pi = %f\n", pi);

// }

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

static long num_steps = 1000000;  // Increased for better timing
double step;

void calculate_pi(int num_threads)
{      
    int i, nthreads; 
    double pi, *sum;
    double start_time, end_time;
    
    sum = (double*) malloc(num_threads * sizeof(double));
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
    
    printf("Threads: %d | Time: %.6f sec | Pi = %.10f\n", 
           num_threads, end_time - start_time, pi);
    
    free(sum);
}

int main()
{
    printf("SPMD Parallel Pi Calculation - Speedup Report\n");
    printf("==============================================\n");
    
    for (int t = 1; t <= 8; t *= 2) {
        calculate_pi(t);
    }
    
    return 0;
}