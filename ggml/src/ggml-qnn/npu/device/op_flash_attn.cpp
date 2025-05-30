
#include "op_flash_attn.hpp"

#include "type_traits.hpp"
#include "util.hpp"

namespace hexagon {

bool flash_attn_f32(tensor * out, compute_params * params) {
    return false;  // TODO: implement flash attention for f32
}

bool is_flash_attn_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                             const npu_device_tensor_spec * srcs, size_t src_len) {
    if (op != NPU_OP_FLASH_ATTN) {
        DEVICE_LOG_DEBUG("op is not NPU_OP_FLASH_ATTN: %d\n", op);
        return false;
    }

    if (!dst || !srcs || src_len < 4) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", op_get_name(op));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst->type));
        return false;
    }

    return false;  // TODO: implement flash attention support check
}

}  // namespace hexagon
