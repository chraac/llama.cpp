#pragma once

#include <memory>

#include "ggml-backend-impl.h"
#include "rpc-mem.hpp"

namespace hexagon {

class npu_buffer {
  public:
    explicit npu_buffer(common::rpc_mem_ptr allocator, size_t size);

    ~npu_buffer() { _allocator->free(_data); }

    bool is_valid() const { return _data != nullptr; }

    void * get_buffer() { return _data; }

    size_t get_size() const { return _size; }

  private:
    common::rpc_mem_ptr _allocator;
    void *              _data = nullptr;
    size_t              _size = 0;

    DISABLE_COPY(npu_buffer);
    DISABLE_MOVE(npu_buffer);
};

class npu_buffer_type : public ggml_backend_buffer_type {
  public:
    explicit npu_buffer_type(ggml_backend_dev_t dev, const std::string & name, common::rpc_mem_ptr rpc_mem);

    const char * get_name() const { return _name.c_str(); }

    size_t get_buffer_alignment() const {
        return 32;  // TODO: get the alignment from device
    }

    size_t get_max_buffer_size() const;

    ggml_backend_buffer_t allocate_buffer(ggml_backend_buffer_type_t buft, size_t size);

  private:
    std::string         _name;
    common::rpc_mem_ptr _rpc_mem;

    DISABLE_COPY(npu_buffer_type);
    DISABLE_MOVE(npu_buffer_type);
};

}  // namespace hexagon
