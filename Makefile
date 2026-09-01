# Makefile for building and running CUDA kernels.
#
#   make          # build all kernels in src/
#   make run      # build and run a single kernel
#   make nsys     # timeline profile the chosen kernel (Nsight Systems)
#   make ncu      # per-kernel hardware profile (Nsight Compute)
#   make clean    # remove build artifacts
#
# Profiling and run targets honor KERNEL, e.g.  make ncu KERNEL=vector_add
# On a shared box, pick a free GPU with GPU=, e.g.  make run GPU=1
# Lighten ncu's impact on shared GPUs with NCU_SET=basic (default: full).
#
#   make sanitize KERNEL=k   # compute-sanitizer memcheck + racecheck
#   make torch-test          # pytest correctness for the torch_ext ops
#   make bench               # run the benchmark scaffold
# Set FASTMATH=1 to append --use_fast_math (A/B the rsqrt path without editing source).

NVCC    ?= nvcc
ARCH    ?= native
NVFLAGS ?= -O2 -std=c++17 -arch=$(ARCH)
FASTMATH ?=
ifeq ($(FASTMATH),1)
NVFLAGS += --use_fast_math
endif
VENV    ?= .venv/bin
# -lineinfo lets ncu attribute stalls to source lines without the -G penalty.
PROFFLAGS ?= -lineinfo
KERNEL  ?= vector_add
# Which GPU run/profile targets use (maps to CUDA_VISIBLE_DEVICES).
GPU     ?= 0
GPU_ENV := CUDA_VISIBLE_DEVICES=$(GPU)
# ncu section set: 'basic' holds the perf lock briefly (kinder on a shared
# GPU); 'full' collects everything but replays kernels many times.
NCU_SET ?= full

NSYS    ?= nsys
NCU     ?= ncu

SRC_DIR := src
BIN_DIR := bin

# One binary per .cu file in src/.
SOURCES := $(wildcard $(SRC_DIR)/*.cu)
TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))
# Rebuild a binary when any shared header changes.
HEADERS := $(wildcard $(SRC_DIR)/*.cuh)

.PHONY: all run nsys ncu sanitize torch-test bench clean

all: $(TARGETS)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PROFFLAGS) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build and run the chosen kernel.
run: $(BIN_DIR)/${KERNEL}
	$(GPU_ENV) ./$(BIN_DIR)/${KERNEL}

# Timeline profile: where does time go (kernels vs. copies vs. gaps)?
nsys: $(BIN_DIR)/${KERNEL}
	$(GPU_ENV) $(NSYS) profile --stats=true --force-overwrite=true \
		-o $(BIN_DIR)/${KERNEL}.nsys ./$(BIN_DIR)/${KERNEL}

# Per-kernel hardware profile: why is this kernel slow? Full section set,
# one launch of the named kernel, skipping the warm-up launch.
ncu: $(BIN_DIR)/${KERNEL}
	$(GPU_ENV) $(NCU) --set $(NCU_SET) -k "regex:${KERNEL}" --launch-skip 1 -c 1 -f \
		-o $(BIN_DIR)/${KERNEL}.ncu ./$(BIN_DIR)/${KERNEL}

# Correctness / race checks — valuable for reduction kernels with shared memory.
sanitize: $(BIN_DIR)/${KERNEL}
	$(GPU_ENV) compute-sanitizer --tool memcheck  ./$(BIN_DIR)/${KERNEL}
	$(GPU_ENV) compute-sanitizer --tool racecheck ./$(BIN_DIR)/${KERNEL}

# PyTorch-extension surface (built out-of-tree by torch.utils.cpp_extension.load).
torch-test:
	$(GPU_ENV) $(VENV)/pytest -q tests/

bench:
	$(GPU_ENV) $(VENV)/python bench/bench_rmsnorm.py

clean:
	rm -rf $(BIN_DIR)
