// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Minimal ck_tile:: namespace compatibility shim for CK-free builds.
// Provides just enough types/functions so that mha_fwd.h, mha_bwd.h,
// the V3 ASM source files, and vec_convert.h compile without the full CK
// source tree.

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// ---------------------------------------------------------------------------
// CK-Tile macro compatibility
// ---------------------------------------------------------------------------
#ifndef CK_TILE_HOST_DEVICE
#define CK_TILE_HOST_DEVICE inline __host__ __device__
#endif
#ifndef CK_TILE_DEVICE
#define CK_TILE_DEVICE inline __device__
#endif
#ifndef CK_TILE_HOST_DEVICE_EXTERN
#define CK_TILE_HOST_DEVICE_EXTERN __host__ __device__
#endif
#ifndef CK_TILE_DEVICE_EXTERN
#define CK_TILE_DEVICE_EXTERN __device__
#endif

namespace ck_tile {

// ---------------------------------------------------------------------------
// Type traits
// ---------------------------------------------------------------------------
template<typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// ---------------------------------------------------------------------------
// Numeric type aliases (match CK definitions)
// ---------------------------------------------------------------------------
using index_t      = int32_t;
using long_index_t = int64_t;

// ---------------------------------------------------------------------------
// number<N>  (compile-time integral constant)
// ---------------------------------------------------------------------------
template <int N>
struct number
{
    static constexpr int value = N;
};

// constant<v> — more general compile-time constant (used by warp_sort.h)
template <auto v>
using constant = std::integral_constant<decltype(v), v>;

// bool_constant<B> (used by warp_sort.h)
template <bool B>
using bool_constant = std::bool_constant<B>;

// ---------------------------------------------------------------------------
// Scalar type aliases
// ---------------------------------------------------------------------------
using fp32_t = float;
using fp16_t = _Float16;
using bf16_t = hip_bfloat16;
using half_t      = fp16_t;
using bfloat16_t  = bf16_t;

// Minimal fp8 type (E4M3 FNUZ by default, matches CK's fp8_t = float8_e4m3_t)
struct alignas(1) fp8_t
{
    using raw_type = uint8_t;
    raw_type data;
    __host__ __device__ constexpr fp8_t() : data(0) {}
    __host__ __device__ explicit constexpr fp8_t(raw_type x) : data(x) {}
    __host__ __device__ explicit constexpr operator float() const
    {
        // Simplified E4M3 FNUZ to float (enough for numeric::max())
        if(data == 0) return 0.0f;
        uint8_t sign = (data >> 7) & 1;
        uint8_t exp  = (data >> 3) & 0xF;
        uint8_t mant = data & 0x7;
        float val;
        if(exp == 0)
            val = (mant / 8.0f) * (1.0f / 128.0f); // subnormal, bias=8
        else
            val = (1.0f + mant / 8.0f) * (1.0f / (float)(1 << (8 - exp))); // bias=8
        return sign ? -val : val;
    }
};

// ---------------------------------------------------------------------------
// Vector types (HIP ext_vector_type)
// ---------------------------------------------------------------------------
using fp32x2_t  = float __attribute__((ext_vector_type(2)));
using int16x2_t = int16_t __attribute__((ext_vector_type(2)));
using int8x2_t  = int8_t __attribute__((ext_vector_type(2)));

// ext_vector_t alias (used by topk_softmax_kernels_group.cu)
template <typename T, int N>
using ext_vector_t = T __attribute__((ext_vector_type(N)));

// ---------------------------------------------------------------------------
// bit_cast / type_convert
// ---------------------------------------------------------------------------
template <typename To, typename From>
CK_TILE_HOST_DEVICE To bit_cast(From x)
{
    static_assert(sizeof(To) == sizeof(From), "bit_cast requires same size");
    To result;
    __builtin_memcpy(&result, &x, sizeof(To));
    return result;
}

template <typename To, typename From>
CK_TILE_HOST_DEVICE constexpr To type_convert(From x)
{
    if constexpr(std::is_same_v<To, From>)
        return x;
    else
        return static_cast<To>(x);
}

// ---------------------------------------------------------------------------
// fp8_interpretation and numeric_traits
// ---------------------------------------------------------------------------
enum class fp8_interpretation
{
    E4M3_OCP  = 0,
    E5M2_OCP  = 1,
    E4M3_FNUZ = 2,
    E5M2_FNUZ = 3,
};

template <typename T>
struct numeric_traits
{
    static constexpr int PackedSize = 1;
};

template <>
struct numeric_traits<fp8_t>
{
#ifdef CK_TILE_USE_OCP_FP8
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E4M3_OCP;
#else
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E4M3_FNUZ;
#endif
    static constexpr int PackedSize = 1;
};

template <>
struct numeric_traits<float>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 23;
    static constexpr int bias = 127;
    static constexpr int PackedSize = 1;
};

// ---------------------------------------------------------------------------
// numeric<T> — numeric limits (matches CK's numeric template)
// ---------------------------------------------------------------------------
template <typename T>
struct numeric
{
    CK_TILE_HOST_DEVICE static constexpr T max() { return std::numeric_limits<T>::max(); }
    CK_TILE_HOST_DEVICE static constexpr T min() { return std::numeric_limits<T>::min(); }
    CK_TILE_HOST_DEVICE static constexpr T lowest() { return std::numeric_limits<T>::lowest(); }
};

template <>
struct numeric<fp8_t>
{
    // E4M3 FNUZ max = 240.0, E4M3 OCP max = 448.0
#ifdef CK_TILE_USE_OCP_FP8
    CK_TILE_HOST_DEVICE static constexpr fp32_t max() { return 448.0f; }
#else
    CK_TILE_HOST_DEVICE static constexpr fp32_t max() { return 240.0f; }
#endif
};

// ---------------------------------------------------------------------------
// thread_buffer<T, N> — simplified array container (matches CK interface)
// ---------------------------------------------------------------------------
template<typename T_, index_t N_>
struct thread_buffer
{
    using value_type = remove_cvref_t<T_>;
    static constexpr index_t N = N_;
    value_type data[N];

    CK_TILE_HOST_DEVICE constexpr thread_buffer() : data{} {}
    CK_TILE_HOST_DEVICE constexpr thread_buffer(const value_type& o) : data{}
    {
        for(index_t i = 0; i < N; i++) data[i] = o;
    }

    CK_TILE_HOST_DEVICE static constexpr auto size() { return N; }
    CK_TILE_HOST_DEVICE constexpr const value_type& operator[](index_t i) const { return data[i]; }
    CK_TILE_HOST_DEVICE constexpr value_type& operator[](index_t i) { return data[i]; }
    CK_TILE_HOST_DEVICE constexpr value_type& operator()(index_t i) { return data[i]; }
    CK_TILE_HOST_DEVICE constexpr const value_type& at(index_t i) const { return data[i]; }
    CK_TILE_HOST_DEVICE constexpr value_type& at(index_t i) { return data[i]; }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr value_type& at(number<I>) { return data[I]; }

    template<typename Tx>
    CK_TILE_HOST_DEVICE auto& get_as()
    {
        static_assert(sizeof(value_type) * N % sizeof(Tx) == 0);
        constexpr int vx = sizeof(value_type) * N / sizeof(Tx);
        return reinterpret_cast<thread_buffer<Tx, vx>&>(data);
    }
    template<typename Tx>
    CK_TILE_HOST_DEVICE constexpr auto& get_as() const
    {
        static_assert(sizeof(value_type) * N % sizeof(Tx) == 0);
        constexpr int vx = sizeof(value_type) * N / sizeof(Tx);
        return reinterpret_cast<const thread_buffer<Tx, vx>&>(data);
    }
};

// ---------------------------------------------------------------------------
// vector_traits<T> — type traits for vector types
// ---------------------------------------------------------------------------
template <typename T, typename = void>
struct vector_traits
{
    using scalar_type = T;
    static constexpr index_t vector_size = 1;
};

template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>, std::enable_if_t<!std::is_class_v<T>>>
{
    using scalar_type = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>, std::enable_if_t<std::is_class_v<T>>>
{
    using scalar_type = typename T::value_type;
    static constexpr index_t vector_size = N * vector_traits<T>::vector_size;
};

// vec_t<T,N> alias for thread_buffer (used by topk_softmax, vec_convert)
template <typename T, index_t N>
using vec_t = thread_buffer<T, N>;

// has_same_scalar_type (needed by thread_buffer::_get_as in some paths)
template <typename T, typename U>
struct has_same_scalar_type : std::false_type {};
template <typename T>
struct has_same_scalar_type<T, T> : std::true_type {};

// ---------------------------------------------------------------------------
// static_for — compile-time unrolled loop (used by warp_sort.h)
// ---------------------------------------------------------------------------
template <index_t Begin, index_t End, index_t Step = 1>
struct static_for
{
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F&& f) const
    {
        if constexpr(Begin < End)
        {
            f(number<Begin>{});
            static_for<Begin + Step, End, Step>{}(std::forward<F>(f));
        }
    }
};

// ---------------------------------------------------------------------------
// stream_config  (identical layout to CK host/stream_config.hpp)
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
    // Execute each callable; abort on first HIP error
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
// get_warp_size
// ---------------------------------------------------------------------------
inline constexpr int get_warp_size() { return 64; }

// ---------------------------------------------------------------------------
// log2e_v<T>
// ---------------------------------------------------------------------------
template <typename T>
inline constexpr T log2e_v = static_cast<T>(1.4426950408889634);

// ---------------------------------------------------------------------------
// Simple tuple with .at(number<N>{}) accessor
// Used by make_generic_attention_mask_coordinates_from_lr_window return value
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
make_generic_attention_mask_coordinates_from_lr_window(index_t left_size,
                                                       index_t right_size,
                                                       index_t sink_size,
                                                       index_t y_total,
                                                       index_t x_total,
                                                       bool is_top_left = true)
{
    index_t left_size_tmp  = is_top_left ? y_total - 1 : x_total - 1;
    index_t right_size_tmp = is_top_left ? x_total - 1 : y_total - 1;

    left_size  = left_size < 0 ? left_size_tmp : left_size;
    right_size = right_size < 0 ? right_size_tmp : right_size;

    index_t x_tmp = is_top_left ? 0 : x_total - y_total;
    index_t y_tmp = is_top_left ? 0 : y_total - x_total;

    index_t x = 1 + right_size + x_tmp;
    index_t y = 1 + left_size + y_tmp;

    return make_tuple(y, x, sink_size, y_total, x_total);
}

} // namespace ck_tile
