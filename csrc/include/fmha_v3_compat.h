// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CK FMHA example-layer types vendored for CK-free V3 ASM builds.
// Sources: composable_kernel/example/ck_tile/01_fmha/{mask,bias,quant,fmha_fwd,fmha_bwd}.hpp
//
// These types are ONLY used on the host side for kernel dispatch logic.
// The actual FMHA kernels are pre-compiled ASM .co files loaded via hipModuleLoad.

#pragma once

#include <ostream>
#include <string>
#include <stdexcept>
#include <algorithm>

// ck_tile_shim.h is already included via aiter_hip_common.h before this header

// ===========================================================================
// mask_enum + mask_info  (from CK example/ck_tile/01_fmha/mask.hpp)
// ===========================================================================

// keep this in sync with ck_tile::GenericAttentionMaskEnum
enum class mask_enum
{
    no_mask = 0,
    mask_top_left,
    mask_bottom_right,
    window_generic,
};

struct mask_info
{
    mask_enum type;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t y, x;
    ck_tile::index_t left, right; // FA style SWA left/right
    ck_tile::index_t sink;

    void serialize(std::ostream& os) const
    {
        if(type == mask_enum::no_mask)
            os << "n";
        else if(type == mask_enum::mask_top_left)
            os << "t(" << left << ":" << right << ")";
        else if(type == mask_enum::mask_bottom_right)
            os << "b(" << left << ":" << right << ")";
        else
        {
            os << "g(" << y << ":" << x << ")";
        }
    }

    static mask_info decode(std::string str, ck_tile::index_t seqlen_q, ck_tile::index_t seqlen_k)
    {
        ck_tile::index_t x_total = seqlen_k;
        ck_tile::index_t y_total = seqlen_q;
        mask_info tmp;
        tmp.seqlen_q = seqlen_q;
        tmp.seqlen_k = seqlen_k;
        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            if(t == "xt" || t == "xb")
            {
                // xformer style sliding window attn from top-left
                ck_tile::index_t window_size = std::stoi(v);
                ck_tile::index_t left_size   = -1;
                ck_tile::index_t right_size  = 0;
                ck_tile::index_t sink_size   = 0;
                if(window_size > 0)
                {
                    left_size  = window_size / 2;
                    right_size = window_size - 1 - left_size;
                }
                auto r = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
                    left_size, right_size, sink_size, y_total, x_total, t == "xt");

                tmp.type  = t == "xt" ? mask_enum::mask_top_left : mask_enum::mask_bottom_right;
                tmp.y     = r.at(ck_tile::number<0>{});
                tmp.x     = r.at(ck_tile::number<1>{});
                tmp.left  = left_size;
                tmp.right = right_size;
            }
            else if(t == "t" || t == "b" || t == "g")
            {
                auto found_1 = v.find(",");
                if(found_1 == std::string::npos)
                {
                    throw std::invalid_argument("invalid mask value: " + str);
                }
                tmp.type              = mask_enum::window_generic;
                ck_tile::index_t v0   = atoi(v.substr(0, found_1).c_str());
                auto found_2          = v.find(',', found_1 + 1);
                ck_tile::index_t v1   = 0;
                ck_tile::index_t sink = 0;
                if(t == "t")
                {
                    if(found_2 != std::string::npos)
                    {
                        v1   = atoi(v.substr(found_1 + 1, found_2 - found_1 - 1).c_str());
                        sink = atoi(v.substr(found_2 + 1).c_str());
                    }
                    else
                    {
                        v1   = atoi(v.substr(found_1 + 1).c_str());
                        sink = 0;
                    }
                    tmp.type = mask_enum::mask_top_left;
                    auto r   = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
                        v0, v1, sink, y_total, x_total, true);
                    tmp.y     = r.at(ck_tile::number<0>{});
                    tmp.x     = r.at(ck_tile::number<1>{});
                    tmp.left  = v0;
                    tmp.right = v1;
                    tmp.sink  = sink;
                }
                else if(t == "b")
                {
                    if(found_2 != std::string::npos)
                    {
                        v1   = atoi(v.substr(found_1 + 1, found_2 - found_1 - 1).c_str());
                        sink = atoi(v.substr(found_2 + 1).c_str());
                    }
                    else
                    {
                        v1   = atoi(v.substr(found_1 + 1).c_str());
                        sink = 0;
                    }
                    tmp.type = mask_enum::mask_bottom_right;
                    auto r   = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
                        v0, v1, sink, y_total, x_total, false);
                    tmp.y     = r.at(ck_tile::number<0>{});
                    tmp.x     = r.at(ck_tile::number<1>{});
                    tmp.left  = v0;
                    tmp.right = v1;
                    tmp.sink  = sink;
                }
                else if(t == "g")
                {
                    tmp.type  = mask_enum::window_generic;
                    tmp.y     = v0;
                    tmp.x     = v1;
                    tmp.left  = v0;
                    tmp.right = v1;
                    tmp.sink  = 0;
                }
            }
            else
            {
                throw std::invalid_argument("invalid mask value: " + str);
            }
        }
        else if(str == "0")
        {
            tmp.type = mask_enum::no_mask;
        }
        else if(str == "1" || str == "t")
        {
            tmp.type  = mask_enum::mask_top_left;
            tmp.y     = seqlen_q;
            tmp.x     = 1;
            tmp.left  = -1;
            tmp.right = 0;
            tmp.sink  = 0;
        }
        else if(str == "2" || str == "b")
        {
            tmp.type  = mask_enum::mask_bottom_right;
            tmp.y     = seqlen_q;
            tmp.x     = seqlen_k - seqlen_q + 1;
            tmp.left  = -1;
            tmp.right = 0;
            tmp.sink  = 0;
        }
        else
        {
            throw std::invalid_argument("invalid mask value: " + str);
        }
        return tmp;
    }

    ck_tile::index_t get_unmaskarea() const
    {
        if(type == mask_enum::no_mask)
            return seqlen_q * seqlen_k;
        ck_tile::index_t area = 0;
        for(ck_tile::index_t i_y = 0; i_y < seqlen_q; ++i_y)
        {
            ck_tile::index_t x_start = std::max(-y + i_y + 1, static_cast<ck_tile::index_t>(0));
            ck_tile::index_t x_end   = std::min(i_y + x, seqlen_k);
            if(x_end > x_start)
            {
                area += (x_end - x_start);
            }
        }
        return area;
    }

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi)
    {
        mi.serialize(os);
        return os;
    }
};

// ===========================================================================
// bias_enum  (from CK example/ck_tile/01_fmha/bias.hpp)
// ===========================================================================

// keep sync with BlockAttentionBiasEnum
enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

// ===========================================================================
// quant_scale_enum  (from CK example/ck_tile/01_fmha/quant.hpp)
// ===========================================================================

// keep sync with BlockAttentionQuantScaleEnum
enum class quant_scale_enum
{
    no_scale      = 0,
    pertensor     = 1,
    blockscale    = 2,
    kv_blockscale = 3, // Q per-tensor, K/V per-page block scale
};

// ===========================================================================
// fmha_fwd_traits  (from CK example/ck_tile/01_fmha/fmha_fwd.hpp:1586-1602)
// ===========================================================================

struct fmha_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type;
    bool has_lse;
    bool has_dropout;
    quant_scale_enum qscale_type;
    bool skip_min_seqlen_q = false;
    bool has_sink          = false;
};

// ===========================================================================
// fmha_fwd_splitkv_traits  (from CK fmha_fwd.hpp:1627-1641)
// ===========================================================================

struct fmha_fwd_splitkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type;
    bool has_lse;
    bool do_fp8_static_quant;
    bool has_sink = false;
};

// ===========================================================================
// fmha_batch_prefill_traits  (from CK fmha_fwd.hpp:1658-1665)
// In CK-free mode, KV cache layout enums are replaced with plain ints
// since V3 ASM does not use batch-prefill codepath.
// ===========================================================================

struct fmha_batch_prefill_traits : public fmha_fwd_traits
{
    int kv_memory_layout = 0;
    int kv_lookup_table  = 0;
    int page_size        = 1;
};
