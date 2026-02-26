// Standalone HSACO kernel launcher — zero AITER dependency.
//
// Demonstrates the complete workflow:
//   1. Load a pre-compiled .co (HSACO) file
//   2. Look up a kernel function by name
//   3. Launch it on the GPU
//   4. Verify results
//
// Build:
//   hipcc -o launch_hsaco launch_hsaco.cpp -std=c++17
//
// Run:
//   ./launch_hsaco kernel_vadd.co

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "[HIP ERROR] %s:%d\n  %s\n  -> %s\n",           \
                    __FILE__, __LINE__, #call, hipGetErrorString(err));       \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <path_to_kernel.co> [kernel_name]\n\n"
                "Example:\n"
                "  # Compile a kernel to .co:\n"
                "  hipcc --genco --offload-arch=gfx950 -o kernel_vadd.co kernel_vadd.cpp\n\n"
                "  # Launch it:\n"
                "  %s kernel_vadd.co vector_add\n",
                argv[0], argv[0]);
        return 1;
    }

    const char* co_path     = argv[1];
    const char* kernel_name = (argc > 2) ? argv[2] : "vector_add";
    const int   N           = 1024 * 1024;  // 1M elements

    // ---- Device info ----
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("GPU: %s  arch: %s  CUs: %d\n\n",
           props.name, props.gcnArchName, props.multiProcessorCount);

    // ---- Step 1: Load the .co file ----
    printf("Step 1: hipModuleLoad(\"%s\")\n", co_path);
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, co_path));
    printf("  OK\n");

    // ---- Step 2: Get kernel function ----
    printf("Step 2: hipModuleGetFunction(\"%s\")\n", kernel_name);
    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, module, kernel_name));
    printf("  OK\n");

    // ---- Step 3: Prepare data ----
    printf("Step 3: Allocating %d floats (%.1f MB)\n", N,
           N * sizeof(float) / (1024.0 * 1024.0));

    std::vector<float> h_A(N), h_B(N), h_C(N, 0.0f);
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i) * 0.5f;
    }

    float *d_A, *d_B, *d_C;
    size_t bytes = N * sizeof(float);
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));
    printf("  OK\n");

    // ---- Step 4: Pack arguments and launch ----
    //
    // For kernels compiled from HIP C++ with extern "C", you can pass
    // arguments as an array of pointers — one pointer per kernel parameter.
    // This is the simplest launch method.
    //
    // For hand-written ASM kernels (like AITER's), you would instead use
    // HIP_LAUNCH_PARAM_BUFFER_POINTER with a packed struct. See the
    // hsaco_topksoftmax.cpp example for that approach.

    unsigned int n = N;
    void* args[] = { &d_A, &d_B, &d_C, &n };

    int block_size = 256;
    int grid_size  = (N + block_size - 1) / block_size;

    printf("Step 4: hipModuleLaunchKernel  grid=%d  block=%d\n",
           grid_size, block_size);

    HIP_CHECK(hipModuleLaunchKernel(
        func,
        grid_size, 1, 1,    // grid  (x, y, z)
        block_size, 1, 1,   // block (x, y, z)
        0,                   // shared memory bytes
        nullptr,             // stream (default)
        args,                // kernel arguments
        nullptr              // extra (unused with args)
    ));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  OK\n");

    // ---- Step 5: Verify ----
    printf("Step 5: Verifying results\n");

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        float err = fabsf(h_C[i] - expected);
        if (err > 1e-5f) {
            if (errors < 5)
                printf("  MISMATCH at [%d]: got %.4f, expected %.4f\n",
                       i, h_C[i], expected);
            errors++;
        }
        if (err > max_err) max_err = err;
    }

    printf("\n=== Results ===\n");
    printf("  Elements:  %d\n", N);
    printf("  Errors:    %d\n", errors);
    printf("  Max error: %e\n", max_err);
    printf("  C[0]=%.1f  C[1]=%.1f  C[2]=%.1f  C[3]=%.1f\n",
           h_C[0], h_C[1], h_C[2], h_C[3]);
    printf("\n%s\n", errors == 0 ? "PASSED" : "FAILED");

    // ---- Cleanup ----
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipModuleUnload(module));

    return errors == 0 ? 0 : 1;
}
