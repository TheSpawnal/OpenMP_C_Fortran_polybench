# Makefile for 2D Dynamic Programming OpenMP Benchmark
#
# Usage:
#   make              # Build with default compiler (gcc)
#   make intel        # Build with Intel compiler
#   make clang        # Build with Clang
#   make run          # Build and run with default parameters
#   make test         # Run multiple problem sizes
#   make clean        # Remove built files

# Compiler selection
CC = gcc
ICC = icc
CLANG = clang

# Base compilation flags
CFLAGS_BASE = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

# OpenMP flags per compiler
OPENMP_GCC = -fopenmp
OPENMP_ICC = -qopenmp
OPENMP_CLANG = -fopenmp

# Default target
TARGET = dynprog

# Default problem size and threads
SIZE = 4000
THREADS = $(shell nproc 2>/dev/null || echo 4)

# Compiler-specific targets
.PHONY: all gcc intel clang

all: gcc

gcc: CC = gcc
gcc: OPENMP_FLAGS = $(OPENMP_GCC)
gcc: $(TARGET)

intel: CC = $(ICC)
intel: OPENMP_FLAGS = $(OPENMP_ICC)
intel: CFLAGS_BASE += -xHost
intel: $(TARGET)

clang: CC = $(CLANG)
clang: OPENMP_FLAGS = $(OPENMP_CLANG)
clang: $(TARGET)

# Main build rule
$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS_BASE) $(OPENMP_FLAGS) -o $@ $< $(LDFLAGS)
	@echo "════════════════════════════════════════════════════════════"
	@echo "  Built: $(TARGET)"
	@echo "  Compiler: $(CC)"
	@echo "  Flags: $(CFLAGS_BASE) $(OPENMP_FLAGS)"
	@echo "════════════════════════════════════════════════════════════"

# Run targets
.PHONY: run test bench profile

run: $(TARGET)
	@echo "Running with SIZE=$(SIZE), THREADS=$(THREADS)"
	OMP_NUM_THREADS=$(THREADS) ./$(TARGET) $(SIZE)

test: $(TARGET)
	@echo "════════════════════════════════════════════════════════════"
	@echo "  Testing multiple problem sizes"
	@echo "════════════════════════════════════════════════════════════"
	@for size in 1000 2000 4000 8000; do \
		echo ""; \
		echo "Testing SIZE=$$size with $(THREADS) threads..."; \
		echo "────────────────────────────────────────────────────────────"; \
		OMP_NUM_THREADS=$(THREADS) ./$(TARGET) $$size || exit 1; \
	done

bench: $(TARGET)
	@echo "════════════════════════════════════════════════════════════"
	@echo "  Scaling benchmark: varying thread counts"
	@echo "════════════════════════════════════════════════════════════"
	@for threads in 1 2 4 8 16; do \
		echo ""; \
		echo "Running with $$threads threads..."; \
		echo "────────────────────────────────────────────────────────────"; \
		OMP_NUM_THREADS=$$threads ./$(TARGET) $(SIZE) $$threads || exit 1; \
	done

# Performance profiling
profile: CFLAGS_BASE += -g
profile: $(TARGET)
	@echo "Building with profiling symbols..."
	@echo "Run: perf record -g ./$(TARGET) $(SIZE)"
	@echo "Then: perf report"

# Optimization exploration
.PHONY: opt-levels

opt-levels:
	@echo "════════════════════════════════════════════════════════════"
	@echo "  Testing different optimization levels"
	@echo "════════════════════════════════════════════════════════════"
	@for opt in O0 O1 O2 O3 Ofast; do \
		echo ""; \
		echo "Building with -$$opt..."; \
		$(CC) -$$opt -march=native $(OPENMP_GCC) -o $(TARGET)_$$opt $(TARGET).c $(LDFLAGS); \
		echo "Testing -$$opt:"; \
		time OMP_NUM_THREADS=$(THREADS) ./$(TARGET)_$$opt 2000 | tail -n 5; \
		rm -f $(TARGET)_$$opt; \
	done

# Architecture-specific builds
.PHONY: native avx2 avx512

native: CFLAGS_BASE += -march=native
native: $(TARGET)

avx2: CFLAGS_BASE += -mavx2 -mfma
avx2: $(TARGET)

avx512: CFLAGS_BASE += -mavx512f -mavx512dq
avx512: $(TARGET)

# Debugging builds
.PHONY: debug sanitize

debug: CFLAGS_BASE = -O0 -g -Wall -Wextra
debug: $(TARGET)
	@echo "Built debug version with -O0 -g"

sanitize: CFLAGS_BASE += -g -fsanitize=thread
sanitize: LDFLAGS += -fsanitize=thread
sanitize: $(TARGET)
	@echo "Built with thread sanitizer"
	@echo "Run with: ./$(TARGET) 500"

# Static analysis
.PHONY: analyze

analyze:
	@echo "Running static analysis..."
	$(CC) $(CFLAGS_BASE) $(OPENMP_GCC) -fanalyzer -o /dev/null $(TARGET).c 2>&1 | head -20

# Documentation
.PHONY: docs

docs:
	@echo "════════════════════════════════════════════════════════════"
	@echo "  2D Dynamic Programming Benchmark"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "COMPILATION:"
	@echo "  make              # Default GCC build"
	@echo "  make intel        # Intel compiler build"
	@echo "  make clang        # Clang build"
	@echo "  make debug        # Debug build (-O0 -g)"
	@echo "  make sanitize     # Thread sanitizer build"
	@echo ""
	@echo "EXECUTION:"
	@echo "  make run          # Run with default size ($(SIZE))"
	@echo "  make run SIZE=8000 THREADS=16"
	@echo "  make test         # Test multiple sizes"
	@echo "  make bench        # Thread scaling benchmark"
	@echo ""
	@echo "OPTIMIZATION:"
	@echo "  make native       # Native architecture"
	@echo "  make avx2         # AVX2 instructions"
	@echo "  make avx512       # AVX-512 instructions"
	@echo "  make opt-levels   # Test O0, O1, O2, O3, Ofast"
	@echo ""
	@echo "ENVIRONMENT VARIABLES:"
	@echo "  OMP_NUM_THREADS      # Set thread count"
	@echo "  OMP_PROC_BIND        # Thread binding (spread/close/master)"
	@echo "  OMP_PLACES           # Thread placement (threads/cores/sockets)"
	@echo "  OMP_SCHEDULE         # Default schedule (static/dynamic/guided)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  # Run with 8 threads on 4000x4000 matrix"
	@echo "  OMP_NUM_THREADS=8 ./dynprog 4000"
	@echo ""
	@echo "  # Spread threads across cores"
	@echo "  OMP_PROC_BIND=spread OMP_PLACES=cores ./dynprog 8000"
	@echo ""
	@echo "  # Run all tests with thread affinity"
	@echo "  OMP_PROC_BIND=spread OMP_PLACES=cores make test"
	@echo ""
	@echo "For detailed documentation, see README_dynprog.md"
	@echo "════════════════════════════════════════════════════════════"

# Help target
.PHONY: help
help: docs

# Cleanup
.PHONY: clean distclean

clean:
	rm -f $(TARGET) $(TARGET)_*
	@echo "Cleaned build artifacts"

distclean: clean
	rm -f *.o *.out core.*
	@echo "Deep clean completed"

# Version info
.PHONY: version

version:
	@echo "Compiler versions:"
	@echo "─────────────────────────────────────────────────────────────"
	@echo -n "GCC: "; $(CC) --version | head -1 || echo "not found"
	@echo -n "ICC: "; $(ICC) --version | head -1 2>/dev/null || echo "not found"
	@echo -n "Clang: "; $(CLANG) --version | head -1 2>/dev/null || echo "not found"
	@echo ""
	@echo "OpenMP support:"
	@echo "─────────────────────────────────────────────────────────────"
	@echo "Testing GCC OpenMP..."
	@$(CC) $(OPENMP_GCC) -o /tmp/omp_test -x c - <<< 'int main(){return 0;}' 2>/dev/null && echo "  GCC: ✓ Supported" || echo "  GCC: ✗ Not supported"
	@rm -f /tmp/omp_test

# Installation
PREFIX ?= /usr/local
.PHONY: install uninstall

install: $(TARGET)
	install -d $(PREFIX)/bin
	install -m 755 $(TARGET) $(PREFIX)/bin/
	@echo "Installed to $(PREFIX)/bin/$(TARGET)"

uninstall:
	rm -f $(PREFIX)/bin/$(TARGET)
	@echo "Uninstalled $(PREFIX)/bin/$(TARGET)"

# Continuous integration
.PHONY: ci

ci: clean all test
	@echo "════════════════════════════════════════════════════════════"
	@echo "  CI pipeline completed successfully"
	@echo "════════════════════════════════════════════════════════════"
