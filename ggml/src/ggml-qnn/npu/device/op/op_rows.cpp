#include "op_rows.hpp"

#include "type_traits.hpp"

namespace hexagon {

bool get_rows_f32(tensor * out, compute_params * params) {
    // TODO: implement get_rows
    return false;
}

bool set_rows_f32(tensor * out, compute_params * params) {
    // TODO: implement set_rows
    return false;
}

bool is_rows_supported(const npu_device_tensor_op_spec * op_spec,
                       const npu_device_tensor_spec *    dst,
                       const npu_device_tensor_spec *    srcs,
                       size_t                            src_len) {
    const auto op = op_spec->op;
    if (op != NPU_OP_GET_ROWS && op != NPU_OP_SET_ROWS) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (src_len < 2) {
        DEVICE_LOG_DEBUG("[%s]invalid src_len: %zu\n", hexagon::op_get_name(op), src_len);
        return false;
    }

    const auto & src0 = srcs[0];
    const auto & src1 = srcs[1];
    if (op == NPU_OP_GET_ROWS) {
        if (dst->ne[0] != src0.ne[0]) {
            DEVICE_LOG_DEBUG("[%s]dst.ne[0] and src0.ne[0] not match: %ld vs %ld\n", hexagon::op_get_name(op),
                             (long) dst->ne[0], (long) src0.ne[0]);
            return false;
        }

        if (dst->type != src0.type) {
            DEVICE_LOG_DEBUG("[%s]dst.type and src0.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(dst->type), hexagon::get_type_name(src0.type));
            return false;
        }
    } else {
        // NPU_OP_SET_ROWS
        if (dst->ne[0] != src0.ne[0] || dst->ne[0] != src1.ne[0]) {
            DEVICE_LOG_DEBUG("[%s]dst.ne[0], src0.ne[0] and src1.ne[0] not match: %ld vs %ld vs %ld\n",
                             hexagon::op_get_name(op), (long) dst->ne[0], (long) src0.ne[0], (long) src1.ne[0]);
            return false;
        }

        if (src1.type != NPU_DATA_TYPE_I32 && src1.type != NPU_DATA_TYPE_I64) {
            DEVICE_LOG_DEBUG("[%s]src1.type is not I32 or I64: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(src1.type));
            return false;
        }

        if (dst->type != src0.type && !get_type_traits(dst->type).from_float) {
            DEVICE_LOG_DEBUG("[%s]dst.from_float is null: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(dst->type));
            return false;
        }

        if (dst->type != NPU_DATA_TYPE_F16) {
            // TODO: remove this limitation if needed
            DEVICE_LOG_DEBUG("[%s]dst.type is not F16: %s\n", hexagon::op_get_name(op),
                             hexagon::get_type_name(dst->type));
            return false;
        }
    }

    // TODO: remove this limitation
    return false;
}

bool is_rows_required_sync(npu_device_tensor_op       prev_op,
                           const npu_device_ne_type & prev_ne,
                           npu_device_tensor_op       op,
                           const npu_device_ne_type & ne) {
    // TODO: implement is_rows_required_sync
    return false;
}

}  // namespace hexagon
