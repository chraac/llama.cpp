
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
