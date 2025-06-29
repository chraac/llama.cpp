#pragma once

#include <hexagon_types.h>

#include "op_types.hpp"
#include "tensor.hpp"

namespace hexagon {

bool mul_mat_f32(tensor * out, compute_params * params);
bool is_mul_mat_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                          const npu_device_tensor_spec * srcs, size_t src_len);

}  // namespace hexagon
