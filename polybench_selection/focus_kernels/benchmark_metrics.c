/**
 * benchmark_metrics.c: Implementation of performance monitoring and analysis functions
 */

#include "benchmark_metrics.h"
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <errno.h>

// Initialize performance monitoring system
void init_performance_monitoring() {
    #ifdef USE_PAPI
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI initialization failed\n");
    }
    #endif
}

// Start performance monitoring
void start_monitoring(PerfMetrics* metrics) {
    if (!metrics) return;
    
    // Initialize metrics
    memset(metrics, 0, sizeof(PerfMetrics));
    
    // Record start time
    metrics->wall_time = get_wall_time();
    
    // Get initial memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    metrics->avg_memory_kb = usage.ru_maxrss;
    
    #ifdef USE_PAPI
    // Initialize PAPI counters if available
    init_papi_counters();
    #endif
}

// Stop monitoring and calculate metrics
void stop_monitoring(PerfMetrics* metrics) {
    if (!metrics) return;
    
    // Calculate elapsed time
    metrics->wall_time = get_wall_time() - metrics->wall_time;
    metrics->execution_time = metrics->wall_time;
    
    // Get final memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    metrics->peak_memory_kb = usage.ru_maxrss;
    
    // Calculate CPU time
    metrics->cpu_time = (double)(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
                       (double)(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000000.0;
    
    #ifdef USE_PAPI
    // Read cache counters
    read_cache_counters(metrics);
    #endif
}

// Calculate derived metrics
void calculate_derived_metrics(PerfMetrics* metrics, double serial_time, int num_threads) {
    if (!metrics || serial_time <= 0) return;
    
    // Calculate speedup
    metrics->speedup = serial_time / metrics->execution_time;
    
    // Calculate parallel efficiency
    metrics->parallel_efficiency = (metrics->speedup / num_threads) * 100.0;
    
    // Estimate serial fraction (Amdahl's law)
    if (num_threads > 1 && metrics->speedup > 0) {
        double p = (1.0 - 1.0/metrics->speedup) / (1.0 - 1.0/num_threads);
        metrics->serial_fraction = 1.0 - p;
    }
    
    // Calculate CPU utilization
    if (metrics->wall_time > 0) {
        metrics->cpu_utilization = (metrics->cpu_time / metrics->wall_time) * 100.0;
    }
}

// Calculate memory bandwidth
double estimate_memory_bandwidth(size_t bytes_accessed, double time) {
    if (time <= 0) return 0.0;
    return (bytes_accessed / (1024.0 * 1024.0 * 1024.0)) / time; // GB/s
}

// Calculate statistical measures
void calculate_statistics(double* times, int n, PerfMetrics* metrics) {
    if (!times || !metrics || n <= 0) return;
    
    // Calculate mean
    double sum = 0.0;
    metrics->min_time = times[0];
    metrics->max_time = times[0];
    
    for (int i = 0; i < n; i++) {
        sum += times[i];
        if (times[i] < metrics->min_time) metrics->min_time = times[i];
        if (times[i] > metrics->max_time) metrics->max_time = times[i];
    }
    metrics->mean_time = sum / n;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = times[i] - metrics->mean_time;
        variance += diff * diff;
    }
    metrics->std_dev = sqrt(variance / n);
    
    // Calculate 95% confidence interval
    metrics->confidence_interval_95 = calculate_confidence_interval(
        metrics->mean_time, metrics->std_dev, n);
}

// Calculate 95% confidence interval
double calculate_confidence_interval(double mean, double std_dev, int n) {
    if (n <= 1) return 0.0;
    // Using t-distribution approximation for 95% CI
    double t_value = 1.96; // Approximation for large n
    if (n <= 30) {
        // More accurate t-values for small samples
        double t_values[] = {12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23};
        if (n <= 10) t_value = t_values[n-2];
        else t_value = 2.0; // Approximate for n > 10
    }
    return t_value * std_dev / sqrt(n);
}

// Calculate load imbalance factor
double calculate_load_imbalance(double* thread_times, int num_threads) {
    if (!thread_times || num_threads <= 0) return 0.0;
    
    double max_time = thread_times[0];
    double avg_time = thread_times[0];
    
    for (int i = 1; i < num_threads; i++) {
        avg_time += thread_times[i];
        if (thread_times[i] > max_time) max_time = thread_times[i];
    }
    avg_time /= num_threads;
    
    if (avg_time > 0) {
        return (max_time - avg_time) / avg_time;
    }
    return 0.0;
}

// Estimate synchronization overhead
double estimate_synchronization_overhead(double total_time, double computation_time) {
    if (total_time <= 0) return 0.0;
    return ((total_time - computation_time) / total_time) * 100.0;
}

// Export metrics to JSON format
void export_metrics_json(const char* filename, BenchmarkConfig* config) {
    if (!filename || !config) return;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "{\n");
    fprintf(fp, "  \"benchmark\": \"%s\",\n", config->kernel_name);
    fprintf(fp, "  \"category\": \"%s\",\n", config->category);
    fprintf(fp, "  \"problem_size\": %d,\n", config->current_size);
    fprintf(fp, "  \"strategies\": [\n");
    
    for (int i = 0; i < config->num_strategies; i++) {
        Strategy* s = &config->strategies[i];
        PerfMetrics* m = &s->metrics;
        
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": \"%s\",\n", s->name);
        fprintf(fp, "      \"execution_time\": %.6f,\n", m->execution_time);
        fprintf(fp, "      \"speedup\": %.2f,\n", m->speedup);
        fprintf(fp, "      \"efficiency\": %.2f,\n", m->parallel_efficiency);
        fprintf(fp, "      \"gflops\": %.2f,\n", m->gflops);
        fprintf(fp, "      \"memory_peak_mb\": %.2f,\n", m->peak_memory_kb / 1024.0);
        fprintf(fp, "      \"cpu_utilization\": %.2f,\n", m->cpu_utilization);
        fprintf(fp, "      \"mean_time\": %.6f,\n", m->mean_time);
        fprintf(fp, "      \"std_dev\": %.6f,\n", m->std_dev);
        fprintf(fp, "      \"confidence_interval\": %.6f\n", m->confidence_interval_95);
        fprintf(fp, "    }%s\n", (i < config->num_strategies - 1) ? "," : "");
    }
    
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    
    fclose(fp);
}

// Export metrics to CSV format
void export_metrics_csv(const char* filename, BenchmarkConfig* config) {
    if (!filename || !config) return;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return;
    }
    
    // CSV header
    fprintf(fp, "Benchmark,Strategy,Threads,Time,Speedup,Efficiency,GFLOPS,Memory_MB,CPU_Util\n");
    
    // Data rows
    for (int i = 0; i < config->num_strategies; i++) {
        Strategy* s = &config->strategies[i];
        PerfMetrics* m = &s->metrics;
        
        fprintf(fp, "%s,%s,%d,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                config->kernel_name,
                s->name,
                s->optimal_threads,
                m->execution_time,
                m->speedup,
                m->parallel_efficiency,
                m->gflops,
                m->peak_memory_kb / 1024.0,
                m->cpu_utilization);
    }
    
    fclose(fp);
}

// Generate data for flamegraph visualization
void generate_flamegraph_data(const char* filename, BenchmarkConfig* config) {
    if (!filename || !config) return;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return;
    }
    
    // Generate stack trace format for flamegraph
    for (int i = 0; i < config->num_strategies; i++) {
        Strategy* s = &config->strategies[i];
        PerfMetrics* m = &s->metrics;
        
        // Format: stack_trace count
        fprintf(fp, "%s;%s;parallel_region %.0f\n", 
                config->kernel_name, s->name, 
                m->execution_time * 1000000); // Convert to microseconds
        
        if (m->serial_fraction > 0) {
            fprintf(fp, "%s;%s;serial_fraction %.0f\n",
                    config->kernel_name, s->name,
                    m->serial_fraction * m->execution_time * 1000000);
        }
        
        if (m->synchronization_overhead > 0) {
            fprintf(fp, "%s;%s;synchronization %.0f\n",
                    config->kernel_name, s->name,
                    (m->synchronization_overhead / 100.0) * m->execution_time * 1000000);
        }
    }
    
    fclose(fp);
}

// Compare strategies and print summary
void compare_strategies(BenchmarkConfig* config) {
    if (!config || config->num_strategies == 0) return;
    
    printf("\n=== Strategy Comparison Summary ===\n");
    printf("Benchmark: %s\n", config->kernel_name);
    printf("Problem Size: %d\n\n", config->current_size);
    
    // Find best strategy
    int best_idx = 0;
    double best_time = config->strategies[0].metrics.execution_time;
    
    for (int i = 1; i < config->num_strategies; i++) {
        if (config->strategies[i].metrics.execution_time < best_time) {
            best_time = config->strategies[i].metrics.execution_time;
            best_idx = i;
        }
    }
    
    printf("Best Strategy: %s\n", config->strategies[best_idx].name);
    printf("Best Time: %.4f seconds\n", best_time);
    printf("Best Speedup: %.2fx\n", config->strategies[best_idx].metrics.speedup);
    printf("Best GFLOPS: %.2f\n\n", config->strategies[best_idx].metrics.gflops);
    
    // Ranking table
    printf("Strategy Rankings:\n");
    printf("%-25s %-12s %-12s %-12s\n", "Strategy", "Time (s)", "Speedup", "Efficiency");
    printf("%-25s %-12s %-12s %-12s\n", "--------", "--------", "-------", "----------");
    
    for (int i = 0; i < config->num_strategies; i++) {
        Strategy* s = &config->strategies[i];
        printf("%-25s %-12.4f %-12.2f %-12.1f%%\n",
               s->name,
               s->metrics.execution_time,
               s->metrics.speedup,
               s->metrics.parallel_efficiency);
    }
}

// Generate scaling report
void generate_scaling_report(BenchmarkConfig* config) {
    if (!config) return;
    
    printf("\n=== Scaling Analysis Report ===\n");
    printf("Benchmark: %s\n\n", config->kernel_name);
    
    printf("Strong Scaling Analysis:\n");
    printf("(Fixed problem size, varying thread count)\n");
    printf("%-25s %-10s %-12s %-12s\n", "Strategy", "Threads", "Time (s)", "Efficiency");
    printf("%-25s %-10s %-12s %-12s\n", "--------", "-------", "--------", "----------");
    
    // Analyze each strategy
    for (int i = 0; i < config->num_strategies; i++) {
        Strategy* s = &config->strategies[i];
        
        printf("%-25s %-10d %-12.4f %-12.1f%%\n",
               s->name,
               s->optimal_threads,
               s->metrics.execution_time,
               s->metrics.parallel_efficiency);
        
        // Amdahl's law analysis
        if (s->metrics.serial_fraction > 0) {
            double max_speedup = 1.0 / s->metrics.serial_fraction;
            printf("  -> Serial fraction: %.2f%% (Max speedup: %.1fx)\n",
                   s->metrics.serial_fraction * 100, max_speedup);
        }
        
        // Load balance analysis
        if (s->metrics.load_imbalance_factor > 0.1) {
            printf("  -> Load imbalance detected: %.1f%%\n",
                   s->metrics.load_imbalance_factor * 100);
        }
    }
}

// Pretty print metrics table
void print_metrics_table(BenchmarkConfig* config) {
    if (!config) return;
    
    printf("\n=== Detailed Performance Metrics ===\n");
    printf("%-25s %-12s %-12s %-12s %-12s %-12s\n",
           "Strategy", "Time (s)", "GFLOPS", "Memory (MB)", "CPU%", "Cache Miss");
    printf("%-25s %-12s %-12s %-12s %-12s %-12s\n",
           "--------", "--------", "------", "-----------", "----", "----------");
    
    for (int i = 0; i < config->num_strategies; i++) {
        Strategy* s = &config->strategies[i];
        PerfMetrics* m = &s->metrics;
        
        printf("%-25s %-12.4f %-12.2f %-12.1f %-12.1f ",
               s->name,
               m->execution_time,
               m->gflops,
               m->peak_memory_kb / 1024.0,
               m->cpu_utilization);
        
        #ifdef USE_PAPI
        if (m->cache_misses_l1 > 0) {
            printf("%-12lld", m->cache_misses_l1);
        } else
        #endif
        {
            printf("%-12s", "N/A");
        }
        printf("\n");
    }
}

#ifdef USE_PAPI
// PAPI-specific functions
static int EventSet = PAPI_NULL;

void init_papi_counters() {
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI initialization failed\n");
        return;
    }
    
    // Create event set
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset failed\n");
        return;
    }
    
    // Add cache miss events
    PAPI_add_event(EventSet, PAPI_L1_DCM);  // L1 data cache misses
    PAPI_add_event(EventSet, PAPI_L2_DCM);  // L2 data cache misses
    
    // Start counting
    PAPI_start(EventSet);
}

void read_cache_counters(PerfMetrics* metrics) {
    long long values[2];
    
    // Read counters
    if (PAPI_read(EventSet, values) == PAPI_OK) {
        metrics->cache_misses_l1 = values[0];
        metrics->cache_misses_l2 = values[1];
        
        // Calculate hit ratio (approximate)
        if (metrics->total_operations > 0) {
            metrics->cache_hit_ratio = 1.0 - 
                ((double)metrics->cache_misses_l1 / metrics->total_operations);
        }
    }
    
    // Stop and clean up
    PAPI_stop(EventSet, values);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
}
#endif // USE_PAPI