# CUDA Toolchain & Glibc Incompatibility Resolution (Fedora / Modern Linux)

## 1. Overview & System Environment
When compiling `llama.cpp` with native NVIDIA CUDA hardware acceleration on modern Linux distributions (such as Fedora 41/42/43), multiple compatibility conflicts arise between modern system libraries (`glibc 2.41`), newer host compilers (Clang 20 / GCC 15), and the NVIDIA CUDA Toolkit (CUDA 12.9).

*   **Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM, Compute Capability `sm_86`).
*   **CUDA Toolkit Version:** 12.9 (located at `/usr/local/cuda`).
*   **Host OS:** Fedora Linux (x86_64).

---

## 2. Issues Encountered & Root Causes

### Barrier 1: Missing Environment Paths for `nvcc`
*   **Symptom:** `bash: nvcc: command not found` when attempting compilation.
*   **Root Cause:** CUDA 12.9 was installed under `/usr/local/cuda-12.9`, but the binaries (`nvcc`) and shared libraries (`libcudart.so`, `libcublas.so`) were absent from the active shell's `$PATH` and `$LD_LIBRARY_PATH`.

### Barrier 2: Host Compiler Version Rejection (`host_config.h`)
*   **Symptom:** `error: -- unsupported clang version! clang version must be less than 20 and greater than 3.2`
*   **Root Cause:** NVIDIA's `nvcc` compiler relies on a host C++ compiler for non-device code. NVIDIA includes a static whitelist check in `crt/host_config.h`. Fedora's default Clang package was Clang 20, exceeding the pre-approved whitelist limit (`< 20`).

### Barrier 3: `glibc 2.41` Math Header Signature Collision (`noexcept`)
*   **Symptom:** 
    ```text
    /usr/include/bits/mathcalls.h: error: exception specification is incompatible with that of previous function "cospi"
    math_functions.hpp: error: 'rsqrt' is missing exception specification 'throw()'
    ```
*   **Root Cause:** Modern `glibc` (version 2.40+) adopted the C23 math specification in `/usr/include/bits/mathcalls.h`, marking `sinpi`, `cospi`, and `rsqrt` (and float variants) with `noexcept(true)`. 
    In CUDA 12.9, NVIDIA's internal headers (`math_functions.h` and `math_functions.hpp`) declared and defined these exact same functions **without** `noexcept(true)`. In C++, mismatched exception specifications on the same function prototype are fatal syntax errors.

---

## 3. The Step-by-Step Resolution

### Step 1: Export Environment Variables
Ensure the CUDA compiler and dynamic linkers are globally visible:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



Step 3: Patch NVIDIA CUDA Math Headers
Back up the original headers and patch both the declaration (math_functions.h) and inline definition (math_functions.hpp) files with noexcept(true) using sed:

bash
# 1. Patch Declarations (math_functions.h)
sudo cp /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h.bak
sudo sed -i 's/sinpi(double x);/sinpi(double x) noexcept(true);/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/sinpif(float x);/sinpif(float x) noexcept(true);/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/cospi(double x);/cospi(double x) noexcept(true);/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/cospif(float x);/cospif(float x) noexcept(true);/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h
# 2. Patch Inline Implementations (math_functions.hpp)
sudo cp /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp.bak
sudo sed -i 's/__func__(double rsqrt(const double a))/__func__(double rsqrt(const double a) noexcept(true))/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp
sudo sed -i 's/__func__(double sinpi(double a))/__func__(double sinpi(double a) noexcept(true))/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp
sudo sed -i 's/__func__(double cospi(double a))/__func__(double cospi(double a) noexcept(true))/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp
sudo sed -i 's/__func__(float rsqrtf(const float a))/__func__(float rsqrtf(const float a) noexcept(true))/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp
sudo sed -i 's/__func__(float sinpif(const float a))/__func__(float sinpif(const float a) noexcept(true))/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp
sudo sed -i 's/__func__(float cospif(const float a))/__func__(float cospif(const float a) noexcept(true))/' /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.hpp
4. Compilation & Verification
Clean & Build llama.cpp
bash
cd llama.cpp
rm -rf build
# Configure CMake with CUDA and explicit Clang 19 toolchain
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++-19 \
  -DCMAKE_C_COMPILER=/usr/bin/clang-19 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-19
# Build binary
cmake --build build --config Release -j $(nproc)
cd ..