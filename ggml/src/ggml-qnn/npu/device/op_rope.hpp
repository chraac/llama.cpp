#pragma once

#include "op_types.hpp"

namespace hexagon {

bool rope_f32(tensor * out, compute_params * params);
bool is_rope_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst, const npu_device_tensor_spec * srcs,
                       size_t src_len);

}  // namespace hexagon
