# Makefile for OpenMP PolyBench Benchmark Suite
# Supports multiple compilers and optimization levels

# Compiler selection
CC ?= gcc
# Alternative: CC = icc
# Alternative: CC = clang

# Base flags
CFLAGS_BASE = -fopenmp -march=native -std=c11 -Wall
LDFLAGS = -lm -fopenmp

# Optimization levels
CFLAGS_O3 = $(CFLAGS_BASE) -O3 -funroll-loops -ftree-vectorize
CFLAGS_O2 = $(CFLAGS_BASE) -O2
CFLAGS_DEBUG = $(CFLAGS_BASE) -O0 -g -DDEBUG
CFLAGS_PROFILE = $(CFLAGS_BASE) -O3 -pg -fno-omit-frame-pointer

# Default optimization
CFLAGS ?= $(CFLAGS_O3)

# PAPI support (optional)
ifdef USE_PAPI
    CFLAGS += -DUSE_PAPI
    LDFLAGS += -lpapi
endif

# Problem size selection
ifdef SIZE
    ifeq ($(SIZE),MINI)
        CFLAGS += -DMINI
    else ifeq ($(SIZE),SMALL)
        CFLAGS += -DSMALL
    else ifeq ($(SIZE),MEDIUM)
        CFLAGS += -DMEDIUM
    else ifeq ($(SIZE),LARGE)
        CFLAGS += -DLARGE
    else ifeq ($(SIZE),EXTRALARGE)
        CFLAGS += -DEXTRALARGE
    endif
else
    # Default to STANDARD size
    CFLAGS += -DSTANDARD
endif

# Source files
BENCHMARKS = benchmark_2mm benchmark_3mm benchmark_cholesky \
             benchmark_correlation benchmark_nussinov

METRICS_SRC = benchmark_metrics.c
METRICS_OBJ = benchmark_metrics.o

# Executable targets
TARGETS = $(BENCHMARKS)

# Build rules
all: $(TARGETS)

# Metrics object file
$(METRICS_OBJ): $(METRICS_SRC) benchmark_metrics.h
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for benchmarks
benchmark_%: benchmark_%.c $(METRICS_OBJ) benchmark_metrics.h
	$(CC) $(CFLAGS) $< $(METRICS_OBJ) -o $@ $(LDFLAGS)

# Alternative build configurations
debug: CFLAGS = $(CFLAGS_DEBUG)
debug: clean all

profile: CFLAGS = $(CFLAGS_PROFILE)
profile: clean all

o2: CFLAGS = $(CFLAGS_O2)
o2: clean all

# Size-specific builds
mini: SIZE = MINI
mini: clean all

small: SIZE = SMALL
small: clean all

medium: SIZE = MEDIUM
medium: clean all

large: SIZE = LARGE
large: clean all

xlarge: SIZE = EXTRALARGE
xlarge: clean all

# Intel compiler build
intel:
	$(MAKE) CC=icc CFLAGS="-qopenmp -O3 -xHost -ipo -qopt-report=5"

# Clang compiler build
clang:
	$(MAKE) CC=clang CFLAGS="-fopenmp -O3 -march=native -Rpass=loop-vectorize"

# PAPI-enabled build
papi:
	$(MAKE) USE_PAPI=1

# Run tests with small problem size
test: small
	@echo "=== Running Quick Tests ==="
	@for bench in $(BENCHMARKS); do \
		echo "\nTesting $$bench..."; \
		OMP_NUM_THREADS=4 ./$$bench || exit 1; \
	done
	@echo "\n=== All tests passed ==="

# Run comprehensive benchmark suite
benchmark: medium
	@echo "=== Running Comprehensive Benchmarks ==="
	@./run_benchmarks.sh

# Run with profiling
perf: profile
	@for bench in $(BENCHMARKS); do \
		echo "Profiling $$bench..."; \
		OMP_NUM_THREADS=8 perf record -g ./$$bench; \
		perf report --stdio > $$bench.perf; \
	done

# Generate assembly for analysis
asm:
	@for src in $(BENCHMARKS:=.c); do \
		$(CC) $(CFLAGS) -S -fverbose-asm $$src -o $${src%.c}.s; \
	done

# Check vectorization
vec-report:
	$(MAKE) clean
	$(MAKE) CC=gcc CFLAGS="$(CFLAGS_O3) -fopt-info-vec-all=vectorization.log"
	@echo "Vectorization report saved to vectorization.log"

# Clean build artifacts
clean:
	rm -f $(TARGETS) $(METRICS_OBJ) *.o *.s *.perf *.log
	rm -f gmon.out core
	rm -rf *.dSYM

# Clean everything including results
distclean: clean
	rm -f results/*.csv results/*.json results/*.txt
	rm -f *.png *.svg *.pdf

# Help target
help:
	@echo "OpenMP PolyBench Benchmark Suite - Makefile Usage"
	@echo "================================================="
	@echo ""
	@echo "Basic targets:"
	@echo "  make [all]       - Build all benchmarks with O3 optimization"
	@echo "  make debug       - Build with debug symbols"
	@echo "  make profile     - Build with profiling support"
	@echo "  make test        - Run quick tests with small problem size"
	@echo "  make benchmark   - Run comprehensive benchmark suite"
	@echo "  make clean       - Remove build artifacts"
	@echo ""
	@echo "Size-specific builds:"
	@echo "  make mini        - Build with MINI problem size"
	@echo "  make small       - Build with SMALL problem size"
	@echo "  make medium      - Build with MEDIUM problem size (default)"
	@echo "  make large       - Build with LARGE problem size"
	@echo "  make xlarge      - Build with EXTRALARGE problem size"
	@echo ""
	@echo "Compiler-specific builds:"
	@echo "  make intel       - Build with Intel compiler"
	@echo "  make clang       - Build with Clang compiler"
	@echo ""
	@echo "Analysis targets:"
	@echo "  make asm         - Generate assembly code"
	@echo "  make vec-report  - Generate vectorization report"
	@echo "  make perf        - Run with perf profiling"
	@echo ""
	@echo "Variables:"
	@echo "  CC=<compiler>    - Set compiler (gcc/icc/clang)"
	@echo "  SIZE=<size>      - Set problem size"
	@echo "  USE_PAPI=1       - Enable PAPI support"

.PHONY: all clean distclean test benchmark debug profile o2 \
        mini small medium large xlarge intel clang papi \
        perf asm vec-report help