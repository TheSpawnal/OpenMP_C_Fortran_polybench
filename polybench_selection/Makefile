# Makefile for OpenMP Benchmark Suite
# Compiler settings
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -lm -fopenmp

# Alternative compiler options
# For Intel compiler: CC = icc, CFLAGS += -xHost -qopenmp
# For Clang: CC = clang, add -fopenmp=libomp

# Benchmark targets
BENCHMARKS = benchmark_2mm benchmark_cholesky benchmark_jacobi2d \
             benchmark_correlation benchmark_dynprog

# Build all benchmarks
all: $(BENCHMARKS)

# Individual benchmark compilation rules
benchmark_2mm: benchmark_2mm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built 2mm benchmark"

benchmark_cholesky: benchmark_cholesky.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built Cholesky decomposition benchmark"

benchmark_jacobi2d: benchmark_jacobi2d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built Jacobi-2D stencil benchmark"

benchmark_correlation: benchmark_correlation.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built Correlation matrix benchmark"

benchmark_dynprog: benchmark_dynprog.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built Dynamic programming benchmark"

# Debug builds (with -g flag and no optimization)
debug: CFLAGS = -g -fopenmp -Wall -Wextra
debug: all

# Profile builds (with profiling support)
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: all

# Clean build artifacts
clean:
	rm -f $(BENCHMARKS) *.o *.out

# Run all benchmarks with default sizes
run: all
	@echo "Running all benchmarks..."
	@echo "\n=== 2MM Benchmark ==="
	./benchmark_2mm
	@echo "\n=== Cholesky Benchmark ==="
	./benchmark_cholesky
	@echo "\n=== Jacobi-2D Benchmark ==="
	./benchmark_jacobi2d
	@echo "\n=== Correlation Benchmark ==="
	./benchmark_correlation
	@echo "\n=== Dynamic Programming Benchmark ==="
	./benchmark_dynprog

# Test with small problem sizes
test: all
	@echo "Testing with small problem sizes..."
	./benchmark_2mm 100 100 100 100
	./benchmark_cholesky 200
	./benchmark_jacobi2d 256 100
	./benchmark_correlation 500 100
	./benchmark_dynprog 300 300

# Performance testing with various thread counts
perf-test: all
	@echo "Performance testing with different thread counts..."
	@for threads in 1 2 4 8 16; do \
		echo "\n=== Testing with $$threads threads ==="; \
		export OMP_NUM_THREADS=$$threads; \
		./benchmark_2mm 400 400 400 400; \
	done

.PHONY: all clean run test debug profile perf-test
