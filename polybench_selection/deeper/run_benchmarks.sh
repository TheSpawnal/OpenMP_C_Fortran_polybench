#!/bin/bash

# run_benchmarks.sh - Comprehensive benchmark runner for OpenMP PolyBench suite
# Executes benchmarks with various thread counts and problem sizes
# Collects results and generates reports

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark_results_${TIMESTAMP}.csv"
SUMMARY_FILE="${RESULTS_DIR}/benchmark_summary_${TIMESTAMP}.txt"
JSON_FILE="${RESULTS_DIR}/benchmark_results_${TIMESTAMP}.json"

# Default configuration
DEFAULT_THREADS="1 2 4 8 16"
DEFAULT_SIZES="small medium"
DEFAULT_ITERATIONS=5
WARMUP_RUNS=2

# Benchmarks to run
BENCHMARKS="benchmark_2mm benchmark_3mm benchmark_cholesky benchmark_correlation benchmark_nussinov"

# Parse command line arguments
THREADS="${DEFAULT_THREADS}"
SIZES="${DEFAULT_SIZES}"
ITERATIONS="${DEFAULT_ITERATIONS}"
VERBOSE=0
DRY_RUN=0

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t THREADS    Thread counts to test (default: \"$DEFAULT_THREADS\")"
    echo "  -s SIZES      Problem sizes to test (default: \"$DEFAULT_SIZES\")"
    echo "  -i ITERATIONS Number of iterations per test (default: $DEFAULT_ITERATIONS)"
    echo "  -b BENCHMARKS Specific benchmarks to run (default: all)"
    echo "  -v            Verbose output"
    echo "  -d            Dry run (show what would be executed)"
    echo "  -h            Show this help message"
    echo ""
    echo "Available sizes: mini, small, medium, large, xlarge"
    echo "Available benchmarks: 2mm, 3mm, cholesky, correlation, nussinov"
    echo ""
    echo "Examples:"
    echo "  $0 -t \"2 4 8\" -s \"small medium\""
    echo "  $0 -t \"1 2 4 8 16 32\" -s large -b \"2mm cholesky\""
}

while getopts "t:s:i:b:vdh" opt; do
    case ${opt} in
        t) THREADS="$OPTARG" ;;
        s) SIZES="$OPTARG" ;;
        i) ITERATIONS="$OPTARG" ;;
        b) BENCHMARKS="benchmark_${OPTARG// / benchmark_}" ;;
        v) VERBOSE=1 ;;
        d) DRY_RUN=1 ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# System information gathering
get_system_info() {
    echo "=== System Information ===" | tee -a "${SUMMARY_FILE}"
    echo "Date: $(date)" | tee -a "${SUMMARY_FILE}"
    echo "Hostname: $(hostname)" | tee -a "${SUMMARY_FILE}"
    echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name: *//')" | tee -a "${SUMMARY_FILE}"
    echo "Cores: $(nproc)" | tee -a "${SUMMARY_FILE}"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')" | tee -a "${SUMMARY_FILE}"
    echo "Compiler: $(gcc --version | head -n1)" | tee -a "${SUMMARY_FILE}"
    echo "OpenMP: $(echo | cpp -fopenmp -dM | grep _OPENMP | awk '{print $3}')" | tee -a "${SUMMARY_FILE}"
    echo "" | tee -a "${SUMMARY_FILE}"
}

# Function to run a single benchmark
run_benchmark() {
    local bench=$1
    local size=$2
    local threads=$3
    local iterations=$4
    
    local bench_name=$(echo $bench | sed 's/benchmark_//')
    
    if [ $DRY_RUN -eq 1 ]; then
        echo "[DRY RUN] Would execute: OMP_NUM_THREADS=$threads ./$bench (size: $size, iterations: $iterations)"
        return
    fi
    
    echo -ne "${BLUE}Running ${bench_name} (size: $size, threads: $threads)...${NC}"
    
    # Set environment variables
    export OMP_NUM_THREADS=$threads
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    
    # Warmup runs
    for ((w=1; w<=$WARMUP_RUNS; w++)); do
        ./$bench > /dev/null 2>&1
    done
    
    # Actual benchmark runs
    local total_time=0
    local min_time=999999
    local max_time=0
    
    for ((i=1; i<=$iterations; i++)); do
        if [ $VERBOSE -eq 1 ]; then
            echo "  Iteration $i/$iterations"
        fi
        
        # Run benchmark and capture output
        local output=$(./$bench 2>&1)
        
        # Extract execution time (assuming benchmark outputs time in seconds)
        local exec_time=$(echo "$output" | grep -E "Sequential:|Time:" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        if [ -z "$exec_time" ]; then
            echo -e "${RED}Error: Could not extract execution time${NC}"
            continue
        fi
        
        # Update statistics
        total_time=$(echo "$total_time + $exec_time" | bc -l)
        if (( $(echo "$exec_time < $min_time" | bc -l) )); then
            min_time=$exec_time
        fi
        if (( $(echo "$exec_time > $max_time" | bc -l) )); then
            max_time=$exec_time
        fi
    done
    
    # Calculate average
    local avg_time=$(echo "scale=6; $total_time / $iterations" | bc -l)
    
    echo -e "${GREEN} Done${NC} (Avg: ${avg_time}s)"
    
    # Write results to CSV
    echo "$bench_name,$size,$threads,$avg_time,$min_time,$max_time" >> "${RESULTS_FILE}"
    
    # Return average time for summary
    echo "$avg_time"
}

# Function to build benchmarks for a specific size
build_for_size() {
    local size=$1
    
    echo -e "${YELLOW}Building benchmarks for size: $size${NC}"
    
    if [ $DRY_RUN -eq 1 ]; then
        echo "[DRY RUN] Would execute: make clean && make $size"
        return 0
    fi
    
    make clean > /dev/null 2>&1
    if ! make $size > /dev/null 2>&1; then
        echo -e "${RED}Error: Build failed for size $size${NC}"
        return 1
    fi
    
    return 0
}

# Function to generate performance summary
generate_summary() {
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "=== Performance Summary ===" | tee -a "${SUMMARY_FILE}"
    echo "" | tee -a "${SUMMARY_FILE}"
    
    # Process results for each benchmark
    for bench in $BENCHMARKS; do
        local bench_name=$(echo $bench | sed 's/benchmark_//')
        echo "Benchmark: $bench_name" | tee -a "${SUMMARY_FILE}"
        
        # Find best configuration
        local best_config=$(grep "^$bench_name," "${RESULTS_FILE}" | sort -t',' -k4 -n | head -1)
        if [ ! -z "$best_config" ]; then
            local best_size=$(echo $best_config | cut -d',' -f2)
            local best_threads=$(echo $best_config | cut -d',' -f3)
            local best_time=$(echo $best_config | cut -d',' -f4)
            
            echo "  Best configuration: size=$best_size, threads=$best_threads" | tee -a "${SUMMARY_FILE}"
            echo "  Best time: ${best_time}s" | tee -a "${SUMMARY_FILE}"
            
            # Calculate speedup if sequential time is available
            local seq_time=$(grep "^$bench_name,$best_size,1," "${RESULTS_FILE}" | cut -d',' -f4)
            if [ ! -z "$seq_time" ]; then
                local speedup=$(echo "scale=2; $seq_time / $best_time" | bc -l)
                echo "  Speedup: ${speedup}x" | tee -a "${SUMMARY_FILE}"
            fi
        fi
        echo "" | tee -a "${SUMMARY_FILE}"
    done
}

# Function to generate JSON report
generate_json_report() {
    echo "{" > "${JSON_FILE}"
    echo "  \"timestamp\": \"${TIMESTAMP}\"," >> "${JSON_FILE}"
    echo "  \"system\": {" >> "${JSON_FILE}"
    echo "    \"hostname\": \"$(hostname)\"," >> "${JSON_FILE}"
    echo "    \"cpu\": \"$(lscpu | grep 'Model name' | sed 's/Model name: *//')\"," >> "${JSON_FILE}"
    echo "    \"cores\": $(nproc)," >> "${JSON_FILE}"
    echo "    \"compiler\": \"$(gcc --version | head -n1)\"" >> "${JSON_FILE}"
    echo "  }," >> "${JSON_FILE}"
    echo "  \"results\": [" >> "${JSON_FILE}"
    
    local first=1
    while IFS=',' read -r bench size threads avg min max; do
        if [ "$bench" != "Benchmark" ]; then  # Skip header
            if [ $first -eq 0 ]; then
                echo "," >> "${JSON_FILE}"
            fi
            echo -n "    {\"benchmark\": \"$bench\", \"size\": \"$size\", \"threads\": $threads, \"avg_time\": $avg, \"min_time\": $min, \"max_time\": $max}" >> "${JSON_FILE}"
            first=0
        fi
    done < "${RESULTS_FILE}"
    
    echo "" >> "${JSON_FILE}"
    echo "  ]" >> "${JSON_FILE}"
    echo "}" >> "${JSON_FILE}"
}

# Function to generate performance plots (requires gnuplot)
generate_plots() {
    if ! command -v gnuplot &> /dev/null; then
        echo "gnuplot not found, skipping plot generation"
        return
    fi
    
    echo -e "${YELLOW}Generating performance plots...${NC}"
    
    for bench in $BENCHMARKS; do
        local bench_name=$(echo $bench | sed 's/benchmark_//')
        local plot_file="${RESULTS_DIR}/${bench_name}_scaling_${TIMESTAMP}.png"
        
        # Create gnuplot script
        cat > /tmp/plot_${bench_name}.gnu << EOF
set terminal png size 1024,768
set output '$plot_file'
set title 'Scaling Analysis: $bench_name'
set xlabel 'Number of Threads'
set ylabel 'Execution Time (seconds)'
set grid
set key top right
set style data linespoints

plot \\
EOF
        
        local first=1
        for size in $SIZES; do
            if [ $first -eq 0 ]; then
                echo ", \\" >> /tmp/plot_${bench_name}.gnu
            fi
            echo -n "'< grep \"^$bench_name,$size,\" ${RESULTS_FILE}' using 3:4 title 'Size: $size'" >> /tmp/plot_${bench_name}.gnu
            first=0
        done
        
        echo "" >> /tmp/plot_${bench_name}.gnu
        
        gnuplot /tmp/plot_${bench_name}.gnu
        rm /tmp/plot_${bench_name}.gnu
        
        echo "  Plot saved: $plot_file"
    done
}

# Main execution
main() {
    echo -e "${GREEN}=== OpenMP PolyBench Benchmark Suite ===${NC}"
    echo ""
    
    # Gather system information
    get_system_info
    
    # Write CSV header
    echo "Benchmark,Size,Threads,Avg_Time,Min_Time,Max_Time" > "${RESULTS_FILE}"
    
    # Run benchmarks for each size
    for size in $SIZES; do
        echo ""
        echo -e "${YELLOW}=== Testing problem size: $size ===${NC}"
        
        # Build benchmarks for this size
        if ! build_for_size $size; then
            echo -e "${RED}Skipping size $size due to build error${NC}"
            continue
        fi
        
        # Run each benchmark
        for bench in $BENCHMARKS; do
            if [ ! -x "$bench" ]; then
                echo -e "${RED}Warning: $bench not found or not executable${NC}"
                continue
            fi
            
            # Test with different thread counts
            for threads in $THREADS; do
                run_benchmark "$bench" "$size" "$threads" "$ITERATIONS"
            done
            echo ""
        done
    done
    
    if [ $DRY_RUN -eq 0 ]; then
        # Generate summary
        generate_summary
        
        # Generate JSON report
        generate_json_report
        
        # Generate plots if possible
        generate_plots
        
        echo ""
        echo -e "${GREEN}=== Benchmark Complete ===${NC}"
        echo "Results saved to: ${RESULTS_FILE}"
        echo "Summary saved to: ${SUMMARY_FILE}"
        echo "JSON report saved to: ${JSON_FILE}"
    fi
}

# Run main function
main

# Clean up
unset OMP_NUM_THREADS
unset OMP_PROC_BIND
unset OMP_PLACES