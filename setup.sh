#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# The pip CUDA-toolkit wheels ship libcudart.so.13 but no unversioned
# libcudart.so, so `nvcc ... -lcudart` from torch's JIT build would fall back to
# the system CUDA 11 libcudart and the extension ends up with a second CUDA
# runtime (kernels then fail with cudaErrorInvalidResourceHandle). Add the
# linker-name symlink next to the real lib.
cu13_lib="$(python -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()['purelib'], 'nvidia', 'cu13', 'lib'))")"
if [ -e "$cu13_lib/libcudart.so.13" ] && [ ! -e "$cu13_lib/libcudart.so" ]; then
    ln -s libcudart.so.13 "$cu13_lib/libcudart.so"
fi
