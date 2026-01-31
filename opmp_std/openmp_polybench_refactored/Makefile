# OpenMP PolyBench Benchmark Suite - Makefile
# Aligned with Julia PolyBench implementation

CC ?= gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lm

# Include paths
INCLUDES = -I./include

# Source and object directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

# Common sources
COMMON_SRC = $(SRC_DIR)/metrics.c
COMMON_OBJ = $(OBJ_DIR)/metrics.o

# Benchmark sources
BENCHMARKS = benchmark_2mm benchmark_3mm benchmark_cholesky benchmark_correlation benchmark_nussinov

# Results directory
RESULTS_DIR = results

.PHONY: all clean results benchmarks help test

all: dirs $(BENCHMARKS)

dirs:
	@mkdir -p $(OBJ_DIR) $(RESULTS_DIR)

# Common object file
$(OBJ_DIR)/metrics.o: $(SRC_DIR)/metrics.c include/metrics.h include/benchmark_common.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Pattern rule for benchmarks
benchmark_%: $(SRC_DIR)/benchmark_%.c $(COMMON_OBJ) include/benchmark_common.h include/metrics.h
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

# Individual targets
benchmark_2mm: $(SRC_DIR)/benchmark_2mm.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_3mm: $(SRC_DIR)/benchmark_3mm.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_correlation: $(SRC_DIR)/benchmark_correlation.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

benchmark_nussinov: $(SRC_DIR)/benchmark_nussinov.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(COMMON_OBJ) -o $@ $(LDFLAGS)

# Debug build
debug: CFLAGS = -O0 -g -fopenmp -Wall -Wextra -DDEBUG
debug: all

# Optimized build with specific architecture
native: CFLAGS = -O3 -march=native -mtune=native -fopenmp -ffast-math
native: all

# Build for DAS-5 cluster (Intel Xeon E5-2630-v3)
das5: CFLAGS = -O3 -march=haswell -fopenmp -Wall
das5: all

# Run all benchmarks with LARGE dataset
run: all
	@echo "Running all benchmarks with LARGE dataset..."
	@for bench in $(BENCHMARKS); do \
		echo ""; \
		echo "=== $$bench ==="; \
		./$$bench --dataset LARGE --output csv; \
	done

# Run quick tests with MINI dataset
test: all
	@echo "Running quick tests with MINI dataset..."
	@for bench in $(BENCHMARKS); do \
		echo ""; \
		./$$bench --dataset MINI --iterations 3 --warmup 1; \
	done

# Scaling study (1, 2, 4, 8, 16 threads)
scaling: all
	@echo "Running scaling study..."
	@for bench in $(BENCHMARKS); do \
		echo ""; \
		echo "=== Scaling: $$bench ==="; \
		for t in 1 2 4 8 16; do \
			echo "Threads: $$t"; \
			OMP_NUM_THREADS=$$t ./$$bench --dataset LARGE --threads $$t --iterations 5 --output csv; \
		done; \
	done

# Clean build artifacts
clean:
	rm -f $(BENCHMARKS)
	rm -rf $(OBJ_DIR)
	rm -f *.o

# Deep clean including results
distclean: clean
	rm -rf $(RESULTS_DIR)

# Help
help:
	@echo "OpenMP PolyBench Benchmark Suite"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all benchmarks (default)"
	@echo "  debug    - Build with debug symbols"
	@echo "  native   - Build with native optimizations"
	@echo "  das5     - Build for DAS-5 cluster"
	@echo "  run      - Run all benchmarks with LARGE dataset"
	@echo "  test     - Quick test with MINI dataset"
	@echo "  scaling  - Thread scaling study"
	@echo "  clean    - Remove build artifacts"
	@echo "  distclean- Remove all generated files"
	@echo ""
	@echo "Benchmark options:"
	@echo "  --dataset SIZE    MINI, SMALL, MEDIUM, LARGE, EXTRALARGE"
	@echo "  --iterations N    Number of timed iterations"
	@echo "  --warmup N        Number of warmup iterations"
	@echo "  --threads N       Number of OpenMP threads"
	@echo "  --output csv      Export results to CSV"
	@echo ""
	@echo "Examples:"
	@echo "  ./benchmark_2mm --dataset LARGE --threads 8 --output csv"
	@echo "  OMP_NUM_THREADS=16 ./benchmark_correlation --dataset MEDIUM"
