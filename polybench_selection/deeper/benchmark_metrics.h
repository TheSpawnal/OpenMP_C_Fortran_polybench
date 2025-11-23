

#ifndef BENCHMARK_METRICS_H
#define BENCHMARK_METRICS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>

#ifdef USE_PAPI
#include <papi.h>  // Optional: for hardware counter access
#endif

#define MAX_STRATEGIES 10
#define MAX_THREADS 128
#define WARMUP_ITERATIONS 3
#define MEASUREMENT_ITERATIONS 10

// Enhanced performance metrics structure
typedef struct {
    // Time-based metrics
    double execution_time;
    double cpu_time;
    double wall_time;
    double speedup;
    double parallel_efficiency;
    double serial_fraction;  // Amdahl's law analysis
    
    // Resource utilization
    double cpu_utilization;
    long peak_memory_kb;
    long avg_memory_kb;
    double memory_bandwidth_gb_s;
    
    // Cache performance (if PAPI available)
    long long cache_misses_l1;
    long long cache_misses_l2;
    long long cache_misses_l3;
    double cache_hit_ratio;
    
    // Computation metrics
    double gflops;
    double arithmetic_intensity;  // FLOPS/byte
    long long total_operations;
    
    // Thread-level metrics
    double load_imbalance_factor;
    double synchronization_overhead;
    double thread_efficiency[MAX_THREADS];
    
    // Statistical measures
    double mean_time;
    double std_dev;
    double min_time;
    double max_time;
    double confidence_interval_95;
    
} PerfMetrics;

// Strategy information
typedef struct {
    char name[128];
    char description[256];
    void (*function)(void*, void*);  // Strategy function pointer
    PerfMetrics metrics;
    int optimal_threads;
    double optimal_tile_size;
} Strategy;

// Benchmark configuration
typedef struct {
    char kernel_name[64];
    char category[64];
    int num_strategies;
    Strategy strategies[MAX_STRATEGIES];
    
    // Problem sizes
    int size_mini, size_small, size_medium, size_large, size_xlarge;
    int current_size;
    
    // Hardware info
    int num_cores;
    int cache_sizes[3];  // L1, L2, L3 in KB
    double peak_bandwidth_gb_s;
    
} BenchmarkConfig;

// Forward declarations for functions implemented in benchmark_metrics.c
void start_monitoring(PerfMetrics* metrics);
void stop_monitoring(PerfMetrics* metrics);
void calculate_derived_metrics(PerfMetrics* metrics, double serial_time, int num_threads);
double estimate_memory_bandwidth(size_t bytes_accessed, double time);
void calculate_statistics(double* times, int n, PerfMetrics* metrics);
double calculate_confidence_interval(double mean, double std_dev, int n);
double calculate_load_imbalance(double* thread_times, int num_threads);
double estimate_synchronization_overhead(double total_time, double computation_time);
void export_metrics_json(const char* filename, BenchmarkConfig* config);
void export_metrics_csv(const char* filename, BenchmarkConfig* config);
void generate_flamegraph_data(const char* filename, BenchmarkConfig* config);
void compare_strategies(BenchmarkConfig* config);
void generate_scaling_report(BenchmarkConfig* config);
void print_metrics_table(BenchmarkConfig* config);

// Cache monitoring (requires PAPI)
#ifdef USE_PAPI
void init_papi_counters();
void read_cache_counters(PerfMetrics* metrics);
#endif

// Inline utility functions
static inline long get_current_memory_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // On Linux, this is in KB
}

static inline double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + time.tv_usec * 1e-6;
}

// Warmup function to stabilize CPU frequency
static inline void warmup_cpu() {
    volatile double dummy = 0.0;
    double start = omp_get_wtime();
    while (omp_get_wtime() - start < 0.1) {
        for (int i = 0; i < 1000000; i++) {
            dummy += i * 0.0001;
        }
    }
}

// Function to calculate GFLOPS
static inline double calculate_gflops(long long ops, double time_seconds) {
    return (ops / 1e9) / time_seconds;
}

// Function to calculate arithmetic intensity
static inline double calculate_arithmetic_intensity(long long ops, size_t bytes_accessed) {
    return (double)ops / bytes_accessed;
}

#endif // BENCHMARK_METRICS_H