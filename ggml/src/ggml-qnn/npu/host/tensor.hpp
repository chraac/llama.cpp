#pragma once

#include "common.hpp"
#include "ggml-impl.h"
#include "hexagon_npu.h"
#include "util.hpp"

namespace hexagon {

// TODO: merge this with device tensor?
class host_tensor {
  public:
    static host_tensor * from_ggml_tensor(ggml_tensor * tensor) {
        if (!tensor || !tensor->extra) {
            return nullptr;
        }
        return static_cast<host_tensor *>(tensor->extra);
    }

    explicit host_tensor(ggml_tensor * tensor, int buffer_fd, uint64_t offset, remote_handle64 device_handle) :
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

    ~host_tensor() {
        if (_device_tensor_handle) {
            npu_device_tensor_free(_device_handle, _device_tensor_handle);
        }
    }

    npu_device_tensor_handle_t get_device_tensor_handle() const { return _device_tensor_handle; }

    void set_src(size_t index, host_tensor * src) {
        if (index >= npu_device_MAX_TENSOR_SRC) {
            return;
        }

        npu_device_tensor_set_src(_device_handle, _device_tensor_handle, index, src->get_device_tensor_handle());
    }

    void set_op(ggml_op op) {
        _info.op = op_to_npu_op(op);
        npu_device_tensor_set_op(_device_handle, _device_tensor_handle, _info.op);
    }

    bool is_valid() const { return _device_tensor_handle != 0; }

  private:
    remote_handle64            _device_handle        = 0;
    npu_device_tensor_handle_t _device_tensor_handle = 0;
    npu_device_tensor_info_t   _info                 = {};

    DISABLE_COPY(host_tensor);
    DISABLE_MOVE(host_tensor);
};

}  // namespace hexagon
