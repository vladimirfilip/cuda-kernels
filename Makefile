# Build, test, profile and benchmark the kernels in kernels/.
#
#   make                     # build every standalone driver
#   make run   KERNEL=matmul # build and run one driver (self-checks + times itself)
#   make test                # pytest across all slices
#   make bench KERNEL=x      # benchmark one slice   (make bench-all for every slice)
#   make nsys  KERNEL=x      # timeline profile      (Nsight Systems)
#   make ncu   KERNEL=x      # hardware counters     (Nsight Compute; see docs/profiling.md)
#   make sanitize KERNEL=x   # compute-sanitizer memcheck + racecheck
#   make clean
#
# On a shared box pick a free GPU with GPU=, e.g. make run GPU=1.
# Set FASTMATH=1 to append --use_fast_math (A/B the rsqrt path without editing source).

VENV    ?= .venv/bin

CUDA_TK := /usr/local/cuda
NVCC    ?= $(CUDA_TK)/bin/nvcc
# 'native' needs CUDA >= 11.5; override for cross-compiles, e.g. ARCH=sm_90.
ARCH    ?= native
NVFLAGS ?= -O2 -std=c++17 -arch=$(ARCH)
FASTMATH ?=
ifeq ($(FASTMATH),1)
NVFLAGS += --use_fast_math
endif
# -lineinfo lets ncu attribute stalls to source lines without the -G penalty.
PROFFLAGS ?= -lineinfo
KERNEL  ?= vector_add
# Which GPU run/profile targets use (maps to CUDA_VISIBLE_DEVICES).
GPU     ?= 0
GPU_ENV := CUDA_VISIBLE_DEVICES=$(GPU)
# ncu section set: 'basic' holds the perf lock briefly (kinder on a shared
# GPU); 'full' collects everything but replays kernels many times.
NCU_SET ?= full

# Nsight and compute-sanitizer are not pip-installable; they come from the
# system CUDA install. Override if they live outside PATH.
NSYS      ?= nsys
NCU       ?= ncu
SANITIZER ?= compute-sanitizer

KERNEL_DIR := kernels
BIN_DIR    := bin

# One binary per kernels/<name>/main.cu.
SOURCES := $(wildcard $(KERNEL_DIR)/*/main.cu)
TARGETS := $(patsubst $(KERNEL_DIR)/%/main.cu,$(BIN_DIR)/%,$(SOURCES))
# Rebuild when any shared or per-slice header changes.
HEADERS := $(wildcard $(KERNEL_DIR)/*/*.cuh) $(wildcard $(KERNEL_DIR)/_common/*.cuh)

# Fail early with an actionable message when an external tool is missing,
# instead of letting the recipe die on a cryptic "command not found".
require = @command -v $(1) >/dev/null 2>&1 || { echo "error: '$(1)' not found on PATH -- install it or pass $(2)=/path/to/tool"; exit 1; }

PY_ENV := PATH="$(CURDIR)/$(VENV):$$PATH" CUDA_HOME="$(CUDA_TK)"

.PHONY: all run test bench bench-all nsys ncu sanitize clean

all: $(TARGETS)

$(BIN_DIR)/%: $(KERNEL_DIR)/%/main.cu $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PROFFLAGS) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build and run the chosen driver. Each one self-checks and self-benchmarks.
run: $(BIN_DIR)/$(KERNEL)
	$(GPU_ENV) ./$(BIN_DIR)/$(KERNEL)

# Correctness across every slice (skips cleanly when no GPU is present).
test:
	$(GPU_ENV) $(PY_ENV) $(VENV)/pytest -q $(KERNEL_DIR)/

bench:
	$(GPU_ENV) $(PY_ENV) $(VENV)/python $(KERNEL_DIR)/$(KERNEL)/bench.py

bench-all:
	@for b in $(wildcard $(KERNEL_DIR)/*/bench.py); do \
		echo "=== $$b ==="; $(GPU_ENV) $(PY_ENV) $(VENV)/python $$b || exit 1; \
	done

# Timeline profile: where does time go (kernels vs. copies vs. gaps)?
# Works without elevated privileges, unlike ncu's counters.
nsys: $(BIN_DIR)/$(KERNEL)
	$(call require,$(NSYS),NSYS)
	$(GPU_ENV) $(NSYS) profile --stats=true --force-overwrite=true \
		-o $(BIN_DIR)/$(KERNEL).nsys ./$(BIN_DIR)/$(KERNEL)

# Per-kernel hardware counters: why is this kernel slow? Needs permission to
# read GPU performance counters -- see docs/profiling.md if this errors.
ncu: $(BIN_DIR)/$(KERNEL)
	$(call require,$(NCU),NCU)
	$(GPU_ENV) $(NCU) --set $(NCU_SET) -k "regex:$(KERNEL)" --launch-skip 1 -c 1 -f \
		-o $(BIN_DIR)/$(KERNEL).ncu ./$(BIN_DIR)/$(KERNEL)

# Correctness / race checks -- valuable for reduction kernels with shared memory.
sanitize: $(BIN_DIR)/$(KERNEL)
	$(call require,$(SANITIZER),SANITIZER)
	$(GPU_ENV) $(SANITIZER) --tool memcheck  ./$(BIN_DIR)/$(KERNEL)
	$(GPU_ENV) $(SANITIZER) --tool racecheck ./$(BIN_DIR)/$(KERNEL)

clean:
	rm -rf $(BIN_DIR)
