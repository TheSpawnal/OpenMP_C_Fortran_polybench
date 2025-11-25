/*

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
    printf("%s\t,%d\t,%d\t,%.6f\t,%.10f\n", 
           schedule_name[kind - 1], 
           chunk_size, 
           num_threads,  
           run_time,
           pi);
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
*/

#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <float.h>

static long num_steps = 100000000;
double step;

typedef struct {
    char schedule[16];
    int chunk;
    double time;
} BestConfig;

void test_schedule_runtime(omp_sched_t kind, int chunk_size, int num_threads) {
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;
    
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
    
    const char* schedule_name[] = {"static", "dynamic", "guided", "auto"};
    printf("%s\t,%d\t,%d\t,%.6f\t,%.10f\n", 
           schedule_name[kind - 1], 
           chunk_size, 
           num_threads,  
           run_time,
           pi);
}

BestConfig find_best_schedule(omp_sched_t kind, int chunks[], int num_chunks, int num_threads) {
    BestConfig best;
    best.time = DBL_MAX;
    
    const char* schedule_name[] = {"static", "dynamic", "guided", "auto"};
    int i;
    double x, pi, sum, start_time, run_time;
    
    step = 1.0 / (double) num_steps;
    
    for (int c = 0; c < num_chunks; c++) {
        sum = 0.0;
        omp_set_num_threads(num_threads);
        omp_set_schedule(kind, chunks[c]);
        
        start_time = omp_get_wtime();
        
        #pragma omp parallel private(x) reduction(+:sum)
        {
            #pragma omp for schedule(runtime)
            for (i = 0; i < num_steps; i++) {
                x = (i + 0.5) * step;
                sum += 4.0 / (1.0 + x*x);
            }
        }
        
        run_time = omp_get_wtime() - start_time;
        
        if (run_time < best.time) {
            best.time = run_time;
            best.chunk = chunks[c];
            strcpy(best.schedule, schedule_name[kind - 1]);
        }
    }
    
    return best;
}

int main() {
    int threads[] = {1, 2, 4, 8};
    int chunks[] = {0, 1, 100, 1000, 10000, 100000, 1000000};
    
    printf("schedule,chunk,threads,time,pi\n");
    
    for (int t = 0; t < 4; t++) {
        for (int c = 0; c < 7; c++) {
            test_schedule_runtime(omp_sched_static, chunks[c], threads[t]);
            test_schedule_runtime(omp_sched_dynamic, chunks[c], threads[t]);
            test_schedule_runtime(omp_sched_guided, chunks[c], threads[t]);
            test_schedule_runtime(omp_sched_auto, chunks[c], threads[t]);
        }
    }
    
    printf("\n=== BEST CONFIGURATIONS PER THREAD COUNT ===\n");
    printf("threads,schedule,chunk,time\n");
    
    for (int t = 0; t < 4; t++) {
        BestConfig best_overall;
        best_overall.time = DBL_MAX;
        
        BestConfig best_static = find_best_schedule(omp_sched_static, chunks, 7, threads[t]);
        BestConfig best_dynamic = find_best_schedule(omp_sched_dynamic, chunks, 7, threads[t]);
        BestConfig best_guided = find_best_schedule(omp_sched_guided, chunks, 7, threads[t]);
        BestConfig best_auto = find_best_schedule(omp_sched_auto, chunks, 1, threads[t]);
        
        if (best_static.time < best_overall.time) best_overall = best_static;
        if (best_dynamic.time < best_overall.time) best_overall = best_dynamic;
        if (best_guided.time < best_overall.time) best_overall = best_guided;
        if (best_auto.time < best_overall.time) best_overall = best_auto;
        
        printf("%d,%s,%d,%.6f\n", 
               threads[t], 
               best_overall.schedule, 
               best_overall.chunk, 
               best_overall.time);
    }
    
    return 0;
}