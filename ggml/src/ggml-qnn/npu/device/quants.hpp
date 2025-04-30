
#include "hexagon_npu.h"
#include "util.hpp"

namespace hexagon {

bool init_f16_f32_table(float * table, size_t count);

void dequantize_row_q4_K(const npu_device_block_q4_K * src, float * dst, size_t count, const float * f16_to_f32_table);

}  // namespace hexagon
