#include "ggml-impl.h"
#include "hexagon_npu.h"

namespace hexagon {

inline enum npu_device_tensor_op_e op_to_npu_op(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
            return NPU_OP_MUL_MAT;
        default:
            return NPU_OP_COUNT;
    }
}

}  // namespace hexagon
