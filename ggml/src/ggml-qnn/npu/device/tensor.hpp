#pragma once

#include <HAP_farf.h>
#include <HAP_mem.h>

#include "hexagon_npu.h"

namespace hexagon {

constexpr const size_t kMaxTensorSrc = 2;

class tensor {
  public:
    explicit tensor(const tensor_info & info) noexcept : _info(info) {
        uint64 phy_address  = 0;
        void * mmap_address = nullptr;
        auto   ret          = HAP_mmap_get(buffer_fd, &mmap_address, &phy_address);
        if (ret != AEE_SUCCESS) {
            FARF(FATAL, "Failed to mmap tensor buffer: %d", (int) ret);
            return;
        }

        _data = static_cast<uint8_t *>(mmap_address);
        FARF(HIGH, "mmap tensor buffer: %p, phy_address: %lx", mmap_address, (long) phy_address);
    }

    ~tensor() noexcept {
        auto ret = HAP_mmap_put(_info.buffer_fd);
        if (ret != AEE_SUCCESS) {
            FARF(FATAL, "Failed to unmap tensor buffer: %d", (int) ret);
        }
        FARF(HIGH, "unmap tensor buffer: %p", _data);
    }

    bool set_src(size_t index, tensor * src) {
        if (index >= kMaxTensorSrc) {
            return false;
        }

        _src[index] = src;
        return true;
    }

    tensor * get_src(size_t index) const {
        if (index >= kMaxTensorSrc) {
            return nullptr;
        }

        return _src[index];
    }

    const tensor_info & get_info() const { return _info; }

    const int64_t * get_ne(size_t index) const { return _info.ne[index]; }

    npu_op get_op() const { return _info.op; }

    uint8_t * get_data() const { return _data + _info.offset; }

    bool is_valid() const { return data != nullptr; }

  private:
    tensor_info _info;
    Tensor *    _src[kMaxTensorSrc] = {};
    uint8_t *   _data               = nullptr;

    tensor(const tensor &)             = delete;
    tensor & operator=(const tensor &) = delete;
};

}  // namespace hexagon
