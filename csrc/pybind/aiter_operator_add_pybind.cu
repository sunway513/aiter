// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &aiter_add, "apply for add with transpose and broadcast.");
    m.def("add_", &aiter_add_, "apply for add_ with transpose and broadcast.");
}
