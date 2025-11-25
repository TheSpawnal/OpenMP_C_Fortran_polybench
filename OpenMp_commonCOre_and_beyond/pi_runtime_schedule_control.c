

#include <stdio.h>
#include <omp.h>
#include <string.h>

static long num_steps = 100000000;
double step;

void test_schedule_runtime(omp_sched_t kind, int chunk_size, int num_threads) {
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;
    omp_sched_t actual_kind;
    int actual_chunk;
    
    step = 1.0 / (double) num_steps;
    omp_set_num_threads(num_threads);
    omp_set_schedule(kind, chunk_size);
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel private(x) reduction(+:sum)
    {
        #pragma omp for schedule(runtime)
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x*x);
        }
    }
    
    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    
    omp_get_schedule(&actual_kind, &actual_chunk);
    
    const char* schedule_name[] = {"static", "dynamic", "guided", "auto"};
    printf("%s,%d,%d,%.10f,%.6f\n", 
           schedule_name[kind - 1], 
           chunk_size, 
           num_threads, 
           pi, 
           run_time);
}

int main() {
    int threads[] = {1, 2, 4, 8};
    int chunks[] = {0, 1, 100, 1000, 10000, 100000, 1000000};
    
    printf("schedule,chunk,threads,pi,time\n");
    
    for (int t = 0; t < 4; t++) {
        for (int c = 0; c < 7; c++) {
            test_schedule_runtime(omp_sched_static, chunks[c], threads[t]);
            test_schedule_runtime(omp_sched_dynamic, chunks[c], threads[t]);
            test_schedule_runtime(omp_sched_guided, chunks[c], threads[t]);
            test_schedule_runtime(omp_sched_auto, chunks[c], threads[t]);
        }
    }
    
    return 0;
}
