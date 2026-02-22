// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host-side utilities for FMHA ASM kernels, in the aiter:: namespace.
// These replace the ck_tile:: shim types for CK-free builds so that
// non-CK modules have zero conceptual dependency on CK headers.

#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <variant>
#include <hip/hip_runtime.h>

namespace aiter {

// ---------------------------------------------------------------------------
// stream_config  (HIP stream wrapper, layout-compatible with CK)
// ---------------------------------------------------------------------------
struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 3;
    int nrepeat_           = 10;
    bool is_gpu_timer_     = true;
    bool flush_cache_      = false;
    int rotating_count_    = 1;
};

// ---------------------------------------------------------------------------
// launch_kernel  (simplified: no timing, just execute callables)
// ---------------------------------------------------------------------------
namespace detail {
template <typename... Callables>
inline void launch_and_check(const stream_config& sc, Callables&&... callables)
{
    if(!((static_cast<void>(callables(sc)), hipPeekAtLastError() == hipSuccess) && ...))
    {
        hipError_t err = hipGetLastError();
        printf("[AITER CK-free] launch_kernel error: %s\n", hipGetErrorString(err));
    }
}
} // namespace detail

template <typename... Callables>
inline float launch_kernel(const stream_config& s, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");
    detail::launch_and_check(s, std::forward<Callables>(callables)...);
    return 0;
}

// ---------------------------------------------------------------------------
// get_warp_size  (runtime query, cached)
// ---------------------------------------------------------------------------
inline int get_warp_size()
{
    static int ws = []() {
        int val = 0;
        hipDevice_t dev;
        (void)hipGetDevice(&dev);
        (void)hipDeviceGetAttribute(&val, hipDeviceAttributeWarpSize, dev);
        return val ? val : 64;
    }();
    return ws;
}

// ---------------------------------------------------------------------------
// log2e_v<T>
// ---------------------------------------------------------------------------
template <typename T>
inline constexpr T log2e_v = static_cast<T>(1.4426950408889634);

// ---------------------------------------------------------------------------
// number<N>  (compile-time integral constant)
// ---------------------------------------------------------------------------
template <int N>
struct number
{
    static constexpr int value = N;
};

// ---------------------------------------------------------------------------
// Simple tuple with .at(number<N>{}) accessor
// ---------------------------------------------------------------------------
template <typename... Ts>
struct simple_tuple
{
    std::tuple<Ts...> data;

    template <int N>
    constexpr auto at(number<N>) const { return std::get<N>(data); }
};

template <typename... Ts>
constexpr auto make_tuple(Ts... args)
{
    return simple_tuple<Ts...>{std::tuple<Ts...>(args...)};
}

// ---------------------------------------------------------------------------
// make_generic_attention_mask_coordinates_from_lr_window
// Vendored from CK block_masking.hpp:752-773 (pure host-side math)
// ---------------------------------------------------------------------------
inline constexpr auto
make_generic_attention_mask_coordinates_from_lr_window(int32_t left_size,
                                                       int32_t right_size,
                                                       int32_t sink_size,
                                                       int32_t y_total,
                                                       int32_t x_total,
                                                       bool is_top_left = true)
{
    int32_t left_size_tmp  = is_top_left ? y_total - 1 : x_total - 1;
    int32_t right_size_tmp = is_top_left ? x_total - 1 : y_total - 1;

    left_size  = left_size < 0 ? left_size_tmp : left_size;
    right_size = right_size < 0 ? right_size_tmp : right_size;

    int32_t x_tmp = is_top_left ? 0 : x_total - y_total;
    int32_t y_tmp = is_top_left ? 0 : y_total - x_total;

    int32_t x = 1 + right_size + x_tmp;
    int32_t y = 1 + left_size + y_tmp;

    return make_tuple(y, x, sink_size, y_total, x_total);
}

} // namespace aiter
