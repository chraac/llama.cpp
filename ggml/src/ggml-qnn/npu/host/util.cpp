#include "util.hpp"

#include <remote.h>

namespace hexagon {

enum npu_device_tensor_op_e op_to_npu_op(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
            return NPU_OP_MUL_MAT;
        default:
            return NPU_OP_COUNT;
    }
}

hexagon_dsp_arch get_dsp_arch(common::rpc_interface_ptr rpc_interface, uint32_t domain_id) {
    if (!rpc_interface || !rpc_interface->is_valid()) {
        return NONE;
    }

    remote_dsp_capability dsp_caps = {};
    dsp_caps.domain                = domain_id;
    dsp_caps.attribute_ID          = ARCH_VER;
    auto ret = rpc_interface->remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_caps, sizeof(dsp_caps));
    if (ret != AEE_SUCCESS) {
        LOG_ERROR("failed to get DSP arch, ret: %d\n", ret);
        return NONE;
    }

    switch (dsp_caps.capability) {
        case V68:
            return V68;
        case V69:
            return V69;
        case V73:
            return V73;
        case V75:
            return V75;
        case V79:
            return V79;
        default:
            LOG_ERROR("unknown DSP arch: %d\n", dsp_caps.capability);
            return NONE;
    }
}

}  // namespace hexagon
