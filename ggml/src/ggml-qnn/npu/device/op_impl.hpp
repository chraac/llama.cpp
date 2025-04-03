#pragma once

#include "hexagon_npu.h"
#include "tensor.hpp"

namespace hexagon {

typedef bool (*compute_func_t)(tensor * dst);

compute_func_t get_compute_func(npu_device_tensor_op_e op);

}  // namespace hexagon
