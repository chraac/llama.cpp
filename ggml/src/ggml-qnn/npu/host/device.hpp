#pragma once

#include <memory>
#include <unordered_map>

#include "buffer.hpp"
#include "common.hpp"
#include "ggml-backend-impl.h"
#include "hexagon_npu.h"
#include "rpc-mem.hpp"

namespace hexagon {

class npu_device {
  public:
    explicit npu_device(backend_index_type device) { GGML_UNUSED(device); }

    ~npu_device();

    const char * get_name() const { return _name.c_str(); }

    const char * get_description() const { return "Hexagon NPU"; }

    bool is_device_valid() const;
    bool init_device(ggml_backend_dev_t dev, const char * params);

    ggml_backend_buffer_type_t get_default_buffer_type();
    bool                       support_buft(ggml_backend_buffer_type_t buft) const;

    remote_handle64 get_device_handle() const { return _device_handle; }

  private:
    std::string                      _name = "hexagon-npu";
    std::shared_ptr<common::rpc_mem> _rpc_mem;
    remote_handle64                  _device_handle = 0;
    std::unique_ptr<npu_buffer_type> _default_buffer_type;

    DISABLE_COPY(npu_device);
    DISABLE_MOVE(npu_device);
};

class npu_graph;

class npu_backend : public ggml_backend {
  public:
    explicit npu_backend(npu_device * device);

    ~npu_backend() {}

    const char * get_name() const {
        // TODO: should we use the device name here?
        return _device->get_name();
    }

    ggml_status graph_compute(ggml_cgraph * cgraph);

  private:
    ggml_guid                                                     _guid   = {};
    npu_device *                                                  _device = nullptr;
    std::unordered_map<ggml_cgraph *, std::shared_ptr<npu_graph>> _graph_cache;

    DISABLE_COPY(npu_backend);
    DISABLE_MOVE(npu_backend);
};

}  // namespace hexagon
