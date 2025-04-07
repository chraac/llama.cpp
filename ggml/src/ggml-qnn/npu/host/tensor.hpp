#pragma once

#include "common.hpp"
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

// TODO: merge this with device tensor?
class npu_tensor {
  public:
    explicit npu_tensor(ggml_tensor * tensor, int buffer_fd, uint64_t offset, remote_handle64 device_handle) :
        _device_handle(device_handle) {
        _info.buffer_fd = buffer_fd;
        _info.offset    = offset;
        _info.type      = tensor->type;
        _info.op        = op_to_npu_op(tensor->op);

        static_assert(sizeof(_info.ne) == sizeof(tensor->ne), "tensor ne size mismatch");
        static_assert(sizeof(_info.nb) == sizeof(tensor->nb), "tensor nb size mismatch");
        memcpy(_info.ne, tensor->ne, sizeof(_info.ne));
        memcpy(_info.nb, tensor->nb, sizeof(_info.nb));

        auto status = npu_device_tensor_init(_device_handle, &_info, &_device_tensor_handle);
        if (status != AEE_SUCCESS) {
            LOG_ERROR("Failed to init tensor: %d", (int) status);
            _device_tensor_handle = 0;
        }
    }

    ~npu_tensor() {
        if (_device_tensor_handle) {
            npu_device_tensor_free(_device_handle, _device_tensor_handle);
        }
    }

    bool is_valid() const { return _device_tensor_handle != 0; }

  private:
    remote_handle64            _device_handle        = 0;
    npu_device_tensor_handle_t _device_tensor_handle = 0;
    npu_device_tensor_info_t   _info                 = {};

    DISABLE_COPY(npu_tensor);
    DISABLE_MOVE(npu_tensor);
};

}  // namespace hexagon
