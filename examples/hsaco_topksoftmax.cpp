// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Standalone HSACO kernel launch example: topksoftmax_4x128x4
//
// Demonstrates loading a pre-compiled .co kernel and launching it via
// the HIP runtime API, then verifying results against a CPU reference.
//
// Build:
//   hipcc -o hsaco_topksoftmax hsaco_topksoftmax.cpp -std=c++17
//
// Run:
//   HIP_VISIBLE_DEVICES=0 ./hsaco_topksoftmax /path/to/hsa/gfx950
//
// The kernel computes top-k softmax over a gating matrix:
//   Input:  gating_output [num_tokens, num_experts]  (fp32)
//   Output: topk_weights  [num_tokens, topk]          (fp32)
//           topk_indices  [num_tokens, topk]          (int32)

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HIP error checking
// ---------------------------------------------------------------------------
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "[HIP ERROR] %s:%d  %s -> %s\n",                \
                    __FILE__, __LINE__, #call, hipGetErrorString(err));       \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Padding structs — must match AITER's kernel ABI (aiter_hip_common.h)
//
// Every kernel argument occupies a 16-byte slot:
//   pointer (8B) + p2 (8B)   = 16B
//   uint32  (4B) + p3 (12B)  = 16B
//   float   (4B) + p3 (12B)  = 16B
// ---------------------------------------------------------------------------
struct p3 { unsigned int _p0, _p1, _p2; };  // 12 bytes
struct p2 { unsigned int _p0, _p1; };        //  8 bytes

// ---------------------------------------------------------------------------
// Kernel argument struct — matches csrc/py_itfs_cu/asm_topksoftmax.cu
// ---------------------------------------------------------------------------
struct __attribute__((packed)) KernelArgs {
    void*        ptr_T;          // output: topk_indices  [num_tokens, topk]  int32
    p2           _pad0;
    void*        ptr_W;          // output: topk_weights  [num_tokens, topk]  fp32
    p2           _pad1;
    void*        ptr_A;          // input:  gating_output [num_tokens, num_experts] fp32
    p2           _pad2;
    unsigned int batch;          // num_tokens
    p3           _pad4;
    unsigned int expert;         // num_experts
    p3           _pad5;
    unsigned int topk;           // top-k value
    p3           _pad6;
    unsigned int renormalize;    // 0 or 1
    p3           _pad7;
    unsigned int out_stride;     // topk_weights row stride in bytes
    p3           _pad8;
};

// ---------------------------------------------------------------------------
// CPU reference implementation for verification
// ---------------------------------------------------------------------------
static void cpu_topk_softmax(const float* gating, int num_tokens,
                             int num_experts, int topk, bool renorm,
                             float* ref_weights, int* ref_indices) {
    for (int t = 0; t < num_tokens; t++) {
        const float* row = gating + t * num_experts;

        // Find top-k indices by partial sort
        std::vector<int> idx(num_experts);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int a, int b) { return row[a] > row[b]; });

        // Softmax over the top-k values
        float max_val = row[idx[0]];
        for (int k = 1; k < topk; k++)
            max_val = std::max(max_val, row[idx[k]]);

        float sum = 0.0f;
        std::vector<float> exp_vals(topk);
        for (int k = 0; k < topk; k++) {
            exp_vals[k] = expf(row[idx[k]] - max_val);
            sum += exp_vals[k];
        }

        for (int k = 0; k < topk; k++) {
            ref_indices[t * topk + k] = idx[k];
            ref_weights[t * topk + k] = exp_vals[k] / sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <path_to_hsa_arch_dir>\n"
                "Example: %s hsa/gfx950\n",
                argv[0], argv[0]);
        return 1;
    }

    // ---- Configuration ----
    // These match the CSV row: topksoftmax_4x128x4.co, subm=4, num_experts=128, topk=4
    const char* hsa_dir     = argv[1];
    const char* co_file     = "topksoftmax/topksoftmax_4x128x4.co";
    const char* kernel_name = "_ZN5aiter19topksoftmax_4x128x4E";
    const int   num_tokens  = 64;
    const int   num_experts = 128;
    const int   topk        = 4;
    const int   SUBM        = 4;    // tile size from CSV subm column
    const bool  renorm      = true;

    printf("=== HSACO Kernel Launch Test: topksoftmax_4x128x4 ===\n");
    printf("Tokens=%d  Experts=%d  TopK=%d\n\n", num_tokens, num_experts, topk);

    // ---- Step 0: Print device info ----
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("GPU: %s (arch: %s)\n\n", props.name, props.gcnArchName);

    // ---- Step 1: Load the .co file ----
    std::string co_path = std::string(hsa_dir) + "/" + co_file;
    printf("Step 1: Loading HSACO from %s\n", co_path.c_str());

    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, co_path.c_str()));
    printf("  hipModuleLoad -> OK\n");

    // ---- Step 2: Get kernel function by mangled name ----
    printf("Step 2: Getting function '%s'\n", kernel_name);

    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, module, kernel_name));
    printf("  hipModuleGetFunction -> OK\n");

    // ---- Step 3: Allocate and initialize data ----
    printf("Step 3: Preparing input data\n");

    const size_t gating_bytes  = num_tokens * num_experts * sizeof(float);
    const size_t weights_bytes = num_tokens * topk * sizeof(float);
    const size_t indices_bytes = num_tokens * topk * sizeof(int);

    std::vector<float> h_gating(num_tokens * num_experts);
    std::vector<float> h_weights(num_tokens * topk, 0.0f);
    std::vector<int>   h_indices(num_tokens * topk, 0);

    srand(42);
    for (auto& v : h_gating)
        v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;

    void *d_gating, *d_weights, *d_indices;
    HIP_CHECK(hipMalloc(&d_gating,  gating_bytes));
    HIP_CHECK(hipMalloc(&d_weights, weights_bytes));
    HIP_CHECK(hipMalloc(&d_indices, indices_bytes));
    HIP_CHECK(hipMemcpy(d_gating, h_gating.data(), gating_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_weights, 0, weights_bytes));
    HIP_CHECK(hipMemset(d_indices, 0, indices_bytes));
    printf("  Allocated %zu bytes on GPU\n", gating_bytes + weights_bytes + indices_bytes);

    // ---- Step 4: Pack kernel arguments ----
    printf("Step 4: Packing kernel arguments (struct size = %zu bytes)\n",
           sizeof(KernelArgs));

    KernelArgs args;
    memset(&args, 0, sizeof(args));
    args.ptr_T       = d_indices;
    args.ptr_W       = d_weights;
    args.ptr_A       = d_gating;
    args.batch       = num_tokens;
    args.expert      = num_experts;
    args.topk        = topk;
    args.renormalize = renorm ? 1 : 0;
    args.out_stride  = topk * sizeof(float);  // row stride in bytes

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    // ---- Step 5: Calculate grid and launch ----
    int gdx = (num_tokens + SUBM - 1) / SUBM;
    printf("Step 5: Launching kernel  grid=(%d,1,1)  block=(256,1,1)\n", gdx);

    HIP_CHECK(hipModuleLaunchKernel(
        func,
        gdx, 1, 1,      // grid
        256, 1, 1,       // block (4 wavefronts)
        0,               // shared memory (kernel manages its own)
        nullptr,         // default stream
        nullptr,
        (void**)&config
    ));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  -> OK\n");

    // ---- Step 6: Verify against CPU reference ----
    printf("\nStep 6: Verifying results\n");

    HIP_CHECK(hipMemcpy(h_weights.data(), d_weights, weights_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_indices.data(), d_indices, indices_bytes, hipMemcpyDeviceToHost));

    std::vector<float> ref_weights(num_tokens * topk);
    std::vector<int>   ref_indices(num_tokens * topk);
    cpu_topk_softmax(h_gating.data(), num_tokens, num_experts, topk, renorm,
                     ref_weights.data(), ref_indices.data());

    // Compare top-k index sets (order may differ between GPU and CPU)
    int idx_match = 0;
    int weight_close = 0;
    float max_weight_err = 0.0f;

    for (int t = 0; t < num_tokens; t++) {
        std::vector<int> gpu_set(h_indices.data() + t * topk,
                                 h_indices.data() + t * topk + topk);
        std::vector<int> cpu_set(ref_indices.data() + t * topk,
                                 ref_indices.data() + t * topk + topk);
        std::sort(gpu_set.begin(), gpu_set.end());
        std::sort(cpu_set.begin(), cpu_set.end());
        if (gpu_set == cpu_set) idx_match++;

        for (int k = 0; k < topk; k++) {
            float err = fabsf(h_weights[t * topk + k] - ref_weights[t * topk + k]);
            max_weight_err = std::max(max_weight_err, err);
            if (err < 0.01f) weight_close++;
        }
    }

    printf("\n=== Results ===\n");
    printf("  Index match:    %d / %d tokens\n", idx_match, num_tokens);
    printf("  Weight match:   %d / %d values (abs err < 0.01)\n",
           weight_close, num_tokens * topk);
    printf("  Max weight err: %.6f\n", max_weight_err);

    // Print a few samples
    printf("\n  Sample output (first 3 tokens):\n");
    for (int t = 0; t < 3 && t < num_tokens; t++) {
        printf("  Token %d GPU: indices=[", t);
        for (int k = 0; k < topk; k++)
            printf("%d%s", h_indices[t * topk + k], k < topk - 1 ? "," : "");
        printf("]  weights=[");
        for (int k = 0; k < topk; k++)
            printf("%.4f%s", h_weights[t * topk + k], k < topk - 1 ? "," : "");
        printf("]\n");

        printf("  Token %d REF: indices=[", t);
        for (int k = 0; k < topk; k++)
            printf("%d%s", ref_indices[t * topk + k], k < topk - 1 ? "," : "");
        printf("]  weights=[");
        for (int k = 0; k < topk; k++)
            printf("%.4f%s", ref_weights[t * topk + k], k < topk - 1 ? "," : "");
        printf("]\n");
    }

    bool pass = (idx_match == num_tokens) && (max_weight_err < 0.01f);
    printf("\n%s\n", pass ? "PASSED" : "FAILED");

    // ---- Cleanup ----
    HIP_CHECK(hipFree(d_gating));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipModuleUnload(module));

    return pass ? 0 : 1;
}
