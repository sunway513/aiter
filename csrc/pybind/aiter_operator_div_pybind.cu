// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("div", &aiter_div, "apply for div with transpose and broadcast.");
    m.def("div_", &aiter_div_, "apply for div_ with transpose and broadcast.");
}
