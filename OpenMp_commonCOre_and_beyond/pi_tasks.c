#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
#define MIN_BLK 10000000

double pi_comp(int Nstart, int Nfinish, double step)
{
    int i, iblk;
    double x, sum = 0.0, sum1, sum2;
    
    if (Nfinish - Nstart < MIN_BLK) {
        for (i = Nstart; i < Nfinish; i++) {
            x = (i + 0.5) * step;
            sum = sum + 4.0 / (1.0 + x * x);
        }
    }
    else {
        iblk = Nfinish - Nstart;
        #pragma omp task shared(sum1)
        sum1 = pi_comp(Nstart, Nfinish - iblk/2, step);
        
        #pragma omp task shared(sum2)
        sum2 = pi_comp(Nfinish - iblk/2, Nfinish, step);
        
        #pragma omp taskwait
        sum = sum1 + sum2;
    }
    return sum;
}

int main()
{
    int i, num_threads;
    double step, pi, sum;
    double start_time, run_time;
    
    step = 1.0 / (double)num_steps;
    
    printf("Calculating PI using recursive task-based approach\n");
    printf("Number of steps: %ld\n", num_steps);
    printf("Minimum block size: %d\n\n", MIN_BLK);
    
    // Test with different number of threads
    for (num_threads = 1; num_threads <= 8; num_threads *= 2) {
        sum = 0.0;
        omp_set_num_threads(num_threads);
        
        start_time = omp_get_wtime();
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Running with %d thread(s)...\n", omp_get_num_threads());
                sum = pi_comp(0, num_steps, step);
            }
        }
        
        pi = step * sum;
        run_time = omp_get_wtime() - start_time;
        
        printf("PI = %.15f\n", pi);
        printf("Runtime: %.6f seconds with %d thread(s)\n", run_time, num_threads);
        printf("-----------------------------------\n");
    }
    
    return 0;
}