#include "ggml-impl.h"
#include "hexagon_npu.h"
#include "rpc-interface.hpp"

namespace hexagon {

enum npu_device_tensor_op_e op_to_npu_op(ggml_op op);

// TODO: merge with qcom_htp_arch
enum hexagon_dsp_arch {
    NONE = 0,
    V68  = 68,
    V69  = 69,
    V73  = 73,
    V75  = 75,
    V79  = 79,  // SD 8 Gen 4 (SM8750)
};

hexagon_dsp_arch get_dsp_arch(common::rpc_interface_ptr rpc_interface, uint32_t domain_id);

}  // namespace hexagon
