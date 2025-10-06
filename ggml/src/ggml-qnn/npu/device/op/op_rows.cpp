#include "op_rows.hpp"

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
    // TODO: implement is_rows_supported
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
