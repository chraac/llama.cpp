#pragma once

#include <HAP_mem.h>

#include "hexagon_npu.h"
#include "util.hpp"

namespace hexagon {

constexpr const size_t kMaxTensorSrc = npu_device_MAX_TENSOR_SRC;

class tensor {
  public:
    explicit tensor(const npu_device_tensor_info_t & info) noexcept : _info(info) {
        uint64 phy_address  = 0;
        void * mmap_address = nullptr;
        auto   ret          = HAP_mmap_get(_info.buffer_fd, &mmap_address, &phy_address);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to mmap tensor buffer: %d", (int) ret);
            return;
        }

        _data = static_cast<uint8_t *>(mmap_address);
        DEVICE_LOG_INFO("tensor(%p[%ldx%ldx%ldx%ld]), fd: %d, offset: %zu, mmap_address: %p, phy_address: 0x%lx\n",
                        (void *) this, (long) _info.ne[0], (long) _info.ne[1], (long) _info.ne[2], (long) _info.ne[3],
                        _info.buffer_fd, _info.offset, (void *) mmap_address, phy_address);
    }

    ~tensor() noexcept {
        auto ret = HAP_mmap_put(_info.buffer_fd);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to unmap tensor buffer: %d", (int) ret);
        }

        DEVICE_LOG_INFO("~tensor(%p) fd: %d", (void *) this, _info.buffer_fd);
    }

    bool set_src(size_t index, tensor * src) {
        if (index >= kMaxTensorSrc) {
            return false;
        }

        _src[index] = src;
        return true;
    }

    void set_op(npu_device_tensor_op op) { _info.op = op; }

    tensor * get_src(size_t index) const {
        if (index >= kMaxTensorSrc) {
            return nullptr;
        }

        return _src[index];
    }

    const npu_device_tensor_info_t & get_info() const { return _info; }

    const int64_t get_ne(size_t index) const { return _info.ne[index]; }

    npu_device_tensor_op get_op() const { return _info.op; }

    npu_device_tensor_data_type get_type() const { return _info.type; }

    uint8_t * get_data() const { return _data + _info.offset; }

    bool is_valid() const { return _data != nullptr; }

  private:
    npu_device_tensor_info_t _info;
    tensor *                 _src[kMaxTensorSrc] = {};
    uint8_t *                _data               = nullptr;

    tensor(const tensor &)         = delete;
    void operator=(const tensor &) = delete;
    tensor(tensor &&)              = delete;
    void operator=(tensor &&)      = delete;
};

}  // namespace hexagon
