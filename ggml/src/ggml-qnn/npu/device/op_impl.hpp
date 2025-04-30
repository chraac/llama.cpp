#pragma once

#include "hexagon_npu.h"
#include "tensor.hpp"

namespace hexagon {

typedef bool (*compute_func_type)(tensor * dst, const compute_params * params);
typedef bool (*op_is_supported_func_type)(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

compute_func_type get_compute_func(tensor * dst);

bool support_op(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
