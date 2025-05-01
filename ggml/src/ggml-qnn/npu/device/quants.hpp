
#include "hexagon_npu.h"
#include "tensor.hpp"
#include "util.hpp"

namespace hexagon {

inline bool is_quantized_type(npu_device_tensor_data_type type) {
    return type == NPU_DATA_TYPE_Q4_K || type == NPU_DATA_TYPE_Q4_0 || type == NPU_DATA_TYPE_Q8_0;
}

inline size_t get_dequantized_row_size(tensor * tensor) {
    auto row_elems_count = tensor->get_ne(0);
    return row_elems_count * sizeof(float);  // currently only f32 is supported
}

bool init_f16_f32_table(float * table, size_t count);

void dequantize_row_q4_K(const npu_device_block_q4_K * src, float * dst, size_t count, const float * f16_to_f32_table);

}  // namespace hexagon
