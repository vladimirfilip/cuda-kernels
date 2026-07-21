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

NVCC    ?= nvcc
ARCH    ?= native
NVFLAGS ?= -O2 -arch=$(ARCH)
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

.PHONY: all run nsys ncu clean

all: $(TARGETS)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu | $(BIN_DIR)
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
# limited to one launch of the named kernel to keep overhead sane.
ncu: $(BIN_DIR)/${KERNEL}
	$(GPU_ENV) $(NCU) --set $(NCU_SET) -k ${KERNEL} -c 1 -f \
		-o $(BIN_DIR)/${KERNEL}.ncu ./$(BIN_DIR)/${KERNEL}

clean:
	rm -rf $(BIN_DIR)
