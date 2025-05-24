#pragma once

#include <type_traits>

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
        static_assert(sizeof(npu_device_tensor_config) < 100, "npu_device_tensor_config size too large");

        _info.buffer_fd = buffer_fd;
        _info.offset    = offset;
        _info.type      = type_to_npu_type(tensor->type);
        _info.op        = op_to_npu_op(tensor->op);
        _info.size      = ggml_nbytes(tensor);

        static_assert(DEVICE_TENSOR_MAX_DIMS == GGML_MAX_DIMS, "tensor dimensions mismatch");
        static_assert(sizeof(_info.ne) == sizeof(tensor->ne), "tensor ne size mismatch");
        static_assert(sizeof(_info.nb) == sizeof(tensor->nb), "tensor nb size mismatch");
        memcpy(_info.ne, tensor->ne, sizeof(_info.ne));
        memcpy(_info.nb, tensor->nb, sizeof(_info.nb));

        auto status = npu_device_tensor_init(_device_handle, &_info, &_device_tensor_handle);
        if (status != AEE_SUCCESS) {
            LOG_ERROR("Failed to init tensor: %d", (int) status);
            _device_tensor_handle = 0;
            return;
        }

        tensor->extra = this;
        _ggml_tensor  = tensor;
        LOG_DEBUG("host_tensor(%p), ggml_tensor(%p[%ldx%ldx%ldx%ld], nb[%ld][%ld][%ld][%ld], %s), handle(%p)\n",
                  (void *) this, (void *) tensor, (long) tensor->ne[0], (long) tensor->ne[1], (long) tensor->ne[2],
                  (long) tensor->ne[3], (long) tensor->nb[0], (long) tensor->nb[1], (long) tensor->nb[2],
                  (long) tensor->nb[3], ggml_type_name(tensor->type), (void *) _device_tensor_handle);
    }

    ~host_tensor() {
        LOG_DEBUG("host_tensor(%p) destroy, device_tensor_handle: %p\n", (void *) this, (void *) _device_tensor_handle);
        if (_device_tensor_handle) {
            npu_device_tensor_free(_device_handle, _device_tensor_handle);
            // TODO: figure out why the _ggml_tensor is invalid here
        }
    }

    npu_device_tensor_handle_t get_device_tensor_handle() const { return _device_tensor_handle; }

    void update_params(ggml_tensor * ggml_tensor) {
        static_assert(sizeof(_op_params) <= sizeof(_ggml_tensor->op_params), "device tensor params size mismatch");
        static_assert(DEVICE_TENSOR_MAX_SRC <= GGML_MAX_SRC, "device tensor src size mismatch");

        GGML_ASSERT(ggml_tensor == _ggml_tensor);
        if (!_ggml_tensor) {
            LOG_DEBUG("host_tensor(%p) _ggml_tensor is null\n", (void *) this);
            return;
        }

        bool params_changed = false;
        auto new_op         = op_to_npu_op(_ggml_tensor->op);
        params_changed |= new_op != _info.op;
        _info.op = new_op;

        if (memcmp(_ggml_tensor->op_params, _op_params, sizeof(_op_params)) != 0) {
            params_changed = true;
            memcpy(_op_params, _ggml_tensor->op_params, sizeof(_op_params));
        }

        npu_device_tensor_handle_t src_tensor_handles[DEVICE_TENSOR_MAX_SRC] = {};
        int                        src_count                                 = 0;
        for (size_t j = 0; j < DEVICE_TENSOR_MAX_SRC && _ggml_tensor->src[j]; ++j) {
            auto * src            = host_tensor::from_ggml_tensor(_ggml_tensor->src[j]);
            src_tensor_handles[j] = src->get_device_tensor_handle();
            src_count++;
            LOG_DEBUG("host_tensor(%p) set_src[%zu]: %p\n", (void *) this, j, (void *) src);
        }

        static_assert(std::is_same<decltype(_src_handles), decltype(src_tensor_handles)>::value,
                      "src tensor handles type mismatch");
        if (src_count != _src_count || memcmp(_src_handles, src_tensor_handles, sizeof(_src_handles)) != 0) {
            params_changed = true;
            memcpy(_src_handles, src_tensor_handles, sizeof(_src_handles));
            _src_count = src_count;
        }

        if (params_changed) {
            npu_device_tensor_update_params(_device_handle, _device_tensor_handle, _info.op, _op_params,
                                            DEVICE_TENSOR_MAX_OP_PARAMS, _src_handles, _src_count);
            LOG_DEBUG("host_tensor(%p) update_params, op: %s, params: [%x, %x, %x, %x], src_count: %d\n", (void *) this,
                      ggml_op_desc(_ggml_tensor), (int) _op_params[0], (int) _op_params[1], (int) _op_params[2],
                      (int) _op_params[3], _src_count);
        } else {
            LOG_DEBUG("host_tensor(%p) update_params, no changes, op: %s, params: [%x, %x, %x, %x], src_count: %d\n",
                      (void *) this, ggml_op_desc(_ggml_tensor), (int) _op_params[0], (int) _op_params[1],
                      (int) _op_params[2], (int) _op_params[3], _src_count);
        }
    }

    bool is_valid() const { return _device_tensor_handle != 0; }

  private:
    remote_handle64            _device_handle                          = 0;
    npu_device_tensor_handle_t _device_tensor_handle                   = 0;
    npu_device_tensor_config   _info                                   = {};
    int32_t                    _op_params[DEVICE_TENSOR_MAX_OP_PARAMS] = {};
    npu_device_tensor_handle_t _src_handles[DEVICE_TENSOR_MAX_SRC]     = {};
    int                        _src_count                              = 0;
    ggml_tensor *              _ggml_tensor                            = nullptr;

    DISABLE_COPY(host_tensor);
    DISABLE_MOVE(host_tensor);
};

}  // namespace hexagon
