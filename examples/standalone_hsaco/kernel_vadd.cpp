// A minimal GPU kernel compiled to HSACO (.co) format.
//
// Compile to .co:
//   /opt/rocm/bin/hipcc --genco --offload-arch=gfx950 -o kernel_vadd.co kernel_vadd.cpp
//
// This produces a standalone code object that can be loaded at runtime
// via hipModuleLoad() without any build-time linkage.

#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(const float* A, const float* B,
                                      float* C, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
