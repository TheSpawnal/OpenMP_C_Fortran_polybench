
#!/bin/bash

# OpenMP Benchmark Suite Runner
# Comprehensive performance evaluation script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
THREAD_COUNTS="1 2 4 8 16"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.csv"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Function to print colored headers
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to run a benchmark with different thread counts
run_benchmark_suite() {
    local benchmark=$1
    local args=$2
    local name=$3
    
    echo -e "${YELLOW}Running ${name}...${NC}"
    echo "${name}" >> ${RESULTS_FILE}
    echo "Threads,Strategy,Time(s)" >> ${RESULTS_FILE}
    
    for threads in ${THREAD_COUNTS}; do
        export OMP_NUM_THREADS=${threads}
        echo -e "  Threads: ${threads}"
        
        # Run benchmark and capture output
        output=$(./${benchmark} ${args} 2>&1)
        
        # Parse and save results
        echo "${output}" | grep -E "seconds" | while read -r line; do
            strategy=$(echo ${line} | cut -d':' -f1)
            time=$(echo ${line} | grep -oP '\d+\.\d+(?= seconds)')
            echo "${threads},${strategy},${time}" >> ${RESULTS_FILE}
        done
    done
    echo "" >> ${RESULTS_FILE}
}

# Function to check CPU information
check_system_info() {
    print_header "System Information"
    echo "Hostname: $(hostname)"
    echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2)"
    echo "CPU Cores: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "OpenMP Version: $(echo | cpp -fopenmp -dM | grep -i openmp | head -1)"
    echo "Compiler: $(gcc --version | head -1)"
    echo ""
}

# Function to set optimal thread affinity
set_thread_affinity() {
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    echo -e "${GREEN}Thread affinity set to: bind=close, places=cores${NC}"
}

# Main execution
main() {
    print_header "OpenMP Benchmark Suite"
    
    # Check system
    check_system_info
    
    # Set thread affinity for better performance
    set_thread_affinity
    
    # Compile benchmarks
    print_header "Compiling Benchmarks"
    make clean
    if make all; then
        echo -e "${GREEN}Compilation successful!${NC}\n"
    else
        echo -e "${RED}Compilation failed!${NC}"
        exit 1
    fi
    
    # Initialize results file
    echo "OpenMP Benchmark Results - ${TIMESTAMP}" > ${RESULTS_FILE}
    echo "System: $(hostname) - $(nproc) cores" >> ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}
    
    # Run benchmarks with different configurations
    print_header "Running Benchmarks"
    
    # Small problem sizes for quick testing
    echo -e "\n${BLUE}=== Small Problem Sizes ===${NC}"
    run_benchmark_suite "benchmark_2mm" "200 250 300 350" "2MM (Small)"
    run_benchmark_suite "benchmark_cholesky" "500" "Cholesky (Small)"
    run_benchmark_suite "benchmark_jacobi2d" "512 500" "Jacobi-2D (Small)"
    run_benchmark_suite "benchmark_correlation" "1000 200" "Correlation (Small)"
    run_benchmark_suite "benchmark_dynprog" "500 500" "Dynamic Programming (Small)"
    
    # Medium problem sizes
    echo -e "\n${BLUE}=== Medium Problem Sizes ===${NC}"
    run_benchmark_suite "benchmark_2mm" "400 450 500 550" "2MM (Medium)"
    run_benchmark_suite "benchmark_cholesky" "1000" "Cholesky (Medium)"
    run_benchmark_suite "benchmark_jacobi2d" "1024 1000" "Jacobi-2D (Medium)"
    run_benchmark_suite "benchmark_correlation" "2000 400" "Correlation (Medium)"
    run_benchmark_suite "benchmark_dynprog" "1000 1000" "Dynamic Programming (Medium)"
    
    # Large problem sizes (optional, may take longer)
    if [[ "$1" == "--large" ]]; then
        echo -e "\n${BLUE}=== Large Problem Sizes ===${NC}"
        run_benchmark_suite "benchmark_2mm" "800 900 1000 1100" "2MM (Large)"
        run_benchmark_suite "benchmark_cholesky" "2000" "Cholesky (Large)"
        run_benchmark_suite "benchmark_jacobi2d" "2048 2000" "Jacobi-2D (Large)"
        run_benchmark_suite "benchmark_correlation" "4000 800" "Correlation (Large)"
        run_benchmark_suite "benchmark_dynprog" "2000 2000" "Dynamic Programming (Large)"
    fi
    
    # Generate summary
    print_header "Benchmark Summary"
    echo -e "${GREEN}Results saved to: ${RESULTS_FILE}${NC}"
    echo ""
    
    # Quick analysis
    echo "Performance Summary (Best times for each benchmark):"
    echo "-----------------------------------------------------"
    
    for benchmark in "2MM" "Cholesky" "Jacobi" "Correlation" "Dynamic"; do
        if grep -q "${benchmark}" ${RESULTS_FILE}; then
            best_single=$(grep -A 100 "${benchmark}" ${RESULTS_FILE} | grep "^1," | awk -F',' '{print $3}' | sort -n | head -1)
            best_multi=$(grep -A 100 "${benchmark}" ${RESULTS_FILE} | grep -E "^(8|16)," | awk -F',' '{print $3}' | sort -n | head -1)
            
            if [[ ! -z "${best_single}" ]] && [[ ! -z "${best_multi}" ]]; then
                speedup=$(echo "scale=2; ${best_single} / ${best_multi}" | bc)
                echo "${benchmark}: Single=${best_single}s, Best Multi=${best_multi}s, Speedup=${speedup}x"
            fi
        fi
    done
}

# Parse command line arguments
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --large    Run with large problem sizes (takes longer)"
    echo "  --help     Show this help message"
    echo "  --threads  Specify thread counts (default: '1 2 4 8 16')"
}

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --large)
            LARGE_TESTS="yes"
            shift
            ;;
        --threads)
            THREAD_COUNTS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main benchmark suite
main $LARGE_TESTS
