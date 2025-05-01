#pragma once

#include <cstdint>

#include "hexagon_npu.h"
#include "tensor.hpp"

namespace hexagon {

struct compute_params {
    size_t        tidx;
    size_t        tcnt;
    const float * f16_to_f32_table;
};

typedef bool (*compute_func_type)(tensor * dst, const compute_params * params);
typedef bool (*op_is_supported_func_type)(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
