
#include "op_flash_attn.hpp"

namespace hexagon {

bool flash_attn_f32(tensor * out, compute_params * params) {
    return false;  // TODO: implement flash attention for f32
}

bool is_flash_attn_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                             const npu_device_tensor_spec * srcs, size_t src_len) {
    return false;  // TODO: implement flash attention support check
}

}  // namespace hexagon
