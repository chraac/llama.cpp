#pragma once

#include "op_types.hpp"

namespace hexagon {

bool rope_f32(tensor * out, compute_params * params);
bool is_rope_supported(const npu_device_tensor_op_spec * op_spec,
                       const npu_device_tensor_spec *    dst,
                       const npu_device_tensor_spec *    srcs,
                       size_t                            src_len);
bool is_rope_required_sync(const npu_device_tensor_op op, const npu_device_tensor_op next_op);

}  // namespace hexagon
