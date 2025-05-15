
#include "hexagon_npu.h"
#include "tensor.hpp"
#include "util.hpp"

namespace hexagon {

bool init_f16_f32_table(float * table, size_t count);

typedef void (*dequantize_row_type)(const void * src, float * dst, size_t count, const float * f16_to_f32_table);

struct device_type_traits {
    npu_device_tensor_data_type type;
    const char *                type_name;
    int64_t                     blck_size;
    bool                        is_quantized;
    dequantize_row_type         dequantize_row;
};

const device_type_traits & get_type_traits(npu_device_tensor_data_type type);

inline bool is_quantized_type(npu_device_tensor_data_type type) {
    return get_type_traits(type).is_quantized;
}

inline size_t get_dequantized_row_size(tensor * tensor) {
    if (!is_quantized_type(tensor->get_type())) {
        return tensor->get_nb(1);  // for f32 and f16
    }

    auto row_elems_count = tensor->get_ne(0);
    return row_elems_count * sizeof(float);  // currently only f32 is supported
}

inline const char * get_type_name(npu_device_tensor_data_type type) {
    return get_type_traits(type).type_name;
}

}  // namespace hexagon

// TODO: move this to a common header
#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
namespace hexagon {

inline auto make_scoped_op_perf_timer(tensor * op) {
    auto * src0 = op->get_src(0);
    auto * src1 = op->get_src(1);
    char   buffer[1024];
    if (src1 == nullptr) {
        snprintf(buffer, sizeof(buffer), "[%s][%lldx%lldx%lldx%lld%s]", op_get_name(op->get_op()), src0->get_ne(0),
                 src0->get_ne(1), src0->get_ne(2), src0->get_ne(3), get_type_name(src0->get_type()));
    } else {
        snprintf(buffer, sizeof(buffer), "[%s][%lldx%lldx%lldx%lld%s],[%lldx%lldx%lldx%lld%s]",
                 op_get_name(op->get_op()), src0->get_ne(0), src0->get_ne(1), src0->get_ne(2), src0->get_ne(3),
                 get_type_name(src0->get_type()), src1->get_ne(0), src1->get_ne(1), src1->get_ne(2), src1->get_ne(3),
                 get_type_name(src1->get_type()));
    }
    return npu_scoped_timer<1024>(buffer);
}

}  // namespace hexagon

#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(op) \
        auto __npu_timer_##__LINE__ = hexagon::make_scoped_op_perf_timer(op)
#else
#    define DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(op) ((void) 0)
#endif
