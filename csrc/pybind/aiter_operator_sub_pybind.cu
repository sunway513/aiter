// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sub", &aiter_sub, "apply for sub with transpose and broadcast.");
    m.def("sub_", &aiter_sub_, "apply for sub_ with transpose and broadcast.");
}
