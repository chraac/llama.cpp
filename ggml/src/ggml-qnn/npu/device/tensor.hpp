#pragma once

#include <HAP_farf.h>
#include <HAP_mem.h>

#include "hexagon_npu.h"

namespace hexagon {

constexpr const size_t kMaxTensorSrc = npu_device_MAX_TENSOR_SRC;

class tensor {
  public:
    explicit tensor(const npu_device_tensor_info_t & info) noexcept : _info(info) {
        uint64 phy_address  = 0;
        void * mmap_address = nullptr;
        auto   ret          = HAP_mmap_get(_info.buffer_fd, &mmap_address, &phy_address);
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

    void set_op(npu_device_tensor_op_e op) {
        _info.op = op;
    }

    tensor * get_src(size_t index) const {
        if (index >= kMaxTensorSrc) {
            return nullptr;
        }

        return _src[index];
    }

    const npu_device_tensor_info_t & get_info() const { return _info; }

    const int64_t get_ne(size_t index) const { return _info.ne[index]; }

    npu_device_tensor_op_e get_op() const { return _info.op; }

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
