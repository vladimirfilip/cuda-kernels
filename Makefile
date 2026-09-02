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

VENV    ?= .venv/bin

CUDA_TK := $(shell $(VENV)/python -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()['purelib'], 'nvidia', 'cu13'))")
NVCC    ?= $(CUDA_TK)/bin/nvcc
# 'native' needs CUDA >= 11.5; override for cross-compiles, e.g. ARCH=sm_90.
ARCH    ?= native
NVFLAGS ?= -O2 -std=c++17 -arch=$(ARCH)
# The toolkit wheels put libcudart under lib/ (not the lib64/ nvcc probes by
# default), so point the linker and the runtime loader at it -- otherwise the
# link fails or the binary picks up the stale system CUDA 11 runtime.
NVLDFLAGS ?= -L $(CUDA_TK)/lib -Xlinker -rpath=$(CUDA_TK)/lib
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

# Nsight and compute-sanitizer are not pip-installable; they come from a system
# CUDA install. Override if they live outside PATH, e.g. NCU=/opt/nvidia/.../ncu.
NSYS      ?= nsys
NCU       ?= ncu
SANITIZER ?= compute-sanitizer

SRC_DIR := src
BIN_DIR := bin

# One binary per .cu file in src/.
SOURCES := $(wildcard $(SRC_DIR)/*.cu)
TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))
# Rebuild a binary when any shared header changes.
HEADERS := $(wildcard $(SRC_DIR)/*.cuh)

# Fail early with an actionable message when an external tool is missing,
# instead of letting the recipe die on a cryptic "command not found".
require = @command -v $(1) >/dev/null 2>&1 || { echo "error: '$(1)' not found on PATH -- install it or pass $(2)=/path/to/tool"; exit 1; }

.PHONY: all run nsys ncu sanitize torch-test bench clean

all: $(TARGETS)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PROFFLAGS) $(NVLDFLAGS) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build and run the chosen kernel.
run: $(BIN_DIR)/${KERNEL}
	$(GPU_ENV) ./$(BIN_DIR)/${KERNEL}

# Timeline profile: where does time go (kernels vs. copies vs. gaps)?
nsys: $(BIN_DIR)/${KERNEL}
	$(call require,$(NSYS),NSYS)
	$(GPU_ENV) $(NSYS) profile --stats=true --force-overwrite=true \
		-o $(BIN_DIR)/${KERNEL}.nsys ./$(BIN_DIR)/${KERNEL}

# Per-kernel hardware profile: why is this kernel slow? Full section set,
# one launch of the named kernel, skipping the warm-up launch.
ncu: $(BIN_DIR)/${KERNEL}
	$(call require,$(NCU),NCU)
	$(GPU_ENV) $(NCU) --set $(NCU_SET) -k "regex:${KERNEL}" --launch-skip 1 -c 1 -f \
		-o $(BIN_DIR)/${KERNEL}.ncu ./$(BIN_DIR)/${KERNEL}

# Correctness / race checks — valuable for reduction kernels with shared memory.
sanitize: $(BIN_DIR)/${KERNEL}
	$(call require,$(SANITIZER),SANITIZER)
	$(GPU_ENV) $(SANITIZER) --tool memcheck  ./$(BIN_DIR)/${KERNEL}
	$(GPU_ENV) $(SANITIZER) --tool racecheck ./$(BIN_DIR)/${KERNEL}

# PyTorch-extension surface (built out-of-tree by torch.utils.cpp_extension.load).
# The system nvcc (CUDA 11.5) is too old for the torch wheel (needs CUDA 13 /
# c++20), so point the JIT build at the same venv toolkit used above. PATH also
# needs the venv bin so torch can find `ninja`.
TORCH_ENV := PATH="$(CURDIR)/$(VENV):$$PATH" CUDA_HOME="$(CUDA_TK)"

torch-test:
	$(GPU_ENV) $(TORCH_ENV) $(VENV)/pytest -q tests/

bench:
	$(GPU_ENV) $(TORCH_ENV) $(VENV)/python bench/bench_rmsnorm.py

clean:
	rm -rf $(BIN_DIR)
