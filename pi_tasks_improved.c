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
    int i;
    double step, pi, sum;
    double start_time, run_time;
    double serial_time = 0.0;
    
    step = 1.0 / (double)num_steps;
    
    printf("===========================================\n");
    printf("PI Calculation - Recursive Task Decomposition\n");
    printf("Number of steps: %ld\n", num_steps);
    printf("Minimum block size: %d\n", MIN_BLK);
    printf("===========================================\n\n");
    
    // Test with 1, 2, 4, and 8 threads
    for (i = 1; i <= 8; i *= 2) {
        sum = 0.0;
        omp_set_num_threads(i);
        
        start_time = omp_get_wtime();
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("num_threads = %d", omp_get_num_threads());
                sum = pi_comp(0, num_steps, step);
            }
        }
        
        pi = step * sum;
        run_time = omp_get_wtime() - start_time;
        
        // Save serial time for speedup calculation
        if (i == 1) {
            serial_time = run_time;
        }
        
        printf("\npi is %.15f in %.6f seconds with %d thread(s)", pi, run_time, i);
        
        // Calculate speedup and efficiency
        if (i > 1) {
            double speedup = serial_time / run_time;
            double efficiency = speedup / i * 100.0;
            printf("\n  Speedup: %.2fx, Efficiency: %.1f%%", speedup, efficiency);
        }
        printf("\n-----------------------------------\n");
    }
    
    return 0;
}