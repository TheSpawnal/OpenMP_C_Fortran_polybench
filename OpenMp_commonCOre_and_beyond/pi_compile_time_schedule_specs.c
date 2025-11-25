/*
Uses compile-time schedule specification via pragmas

Compile:
bashgcc -fopenmp -O2 -o pi_schedule pi_schedule.c -lm
./pi_schedule > results.csv

Schedule behaviors:
static: blocks divided evenly at compile time, minimal overhead
dynamic: runtime assignment, handles load imbalance, higher overhead
guided: decreasing chunk sizes, balance between static/dynamic
auto: compiler/runtime decides

The CSV output reveals which schedule conquers your workload.*/

#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

void test_schedule(const char* schedule_type, int chunk_size, int num_threads) {
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;
    
    step = 1.0 / (double) num_steps;
    omp_set_num_threads(num_threads);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel private(x) reduction(+:sum)
    {
        if (chunk_size > 0) {
            if (strcmp(schedule_type, "static") == 0)
                #pragma omp for schedule(static, chunk_size)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
            else if (strcmp(schedule_type, "dynamic") == 0)
                #pragma omp for schedule(dynamic, chunk_size)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
            else if (strcmp(schedule_type, "guided") == 0)
                #pragma omp for schedule(guided, chunk_size)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
        } else {
            if (strcmp(schedule_type, "static") == 0)
                #pragma omp for schedule(static)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
            else if (strcmp(schedule_type, "dynamic") == 0)
                #pragma omp for schedule(dynamic)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
            else if (strcmp(schedule_type, "guided") == 0)
                #pragma omp for schedule(guided)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
            else if (strcmp(schedule_type, "auto") == 0)
                #pragma omp for schedule(auto)
                for (i = 0; i < num_steps; i++) {
                    x = (i + 0.5) * step;
                    sum += 4.0 / (1.0 + x*x);
                }
        }
    }
    
    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    
    printf("%s,%d,%d,%.10f,%.6f\n", 
           schedule_type, 
           chunk_size > 0 ? chunk_size : 0, 
           num_threads, 
           pi, 
           run_time);
}

int main() {
    int threads[] = {1, 2, 4, 8};
    int chunks[] = {1, 100, 1000, 10000, 100000, 1000000};
    
    printf("schedule,chunk,threads,pi,time\n");
    
    for (int t = 0; t < 4; t++) {
        test_schedule("static", 0, threads[t]);
        test_schedule("dynamic", 0, threads[t]);
        test_schedule("guided", 0, threads[t]);
        test_schedule("auto", 0, threads[t]);
        
        for (int c = 0; c < 6; c++) {
            test_schedule("static", chunks[c], threads[t]);
            test_schedule("dynamic", chunks[c], threads[t]);
            test_schedule("guided", chunks[c], threads[t]);
        }
    }
    
    return 0;
}
