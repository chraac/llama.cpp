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
    explicit npu_device(backend_index_type device);

    ~npu_device();

    const char * get_name() const { return _name.c_str(); }

    const char * get_description() const { return "Hexagon NPU"; }

    size_t get_alignment() const;

    uint32_t get_dsp_domain_id() const { return _dsp_domain_id; }

    bool is_device_valid() const;
    bool init_device(ggml_backend_dev_t dev, const char * params);

    ggml_backend_buffer_type_t get_default_buffer_type();

    bool supports_buft(ggml_backend_buffer_type_t buft) const;
    bool supports_op(const ggml_tensor * op);
    bool offload_op(const ggml_tensor * op);

    remote_handle64 get_device_handle() const { return _device_handle; }

  private:
    std::string                      _name = "hexagon-npu";
    common::rpc_interface_ptr        _rpc_interface;
    common::rpc_mem_ptr              _rpc_mem;
    remote_handle64                  _device_handle = 0;
    std::unique_ptr<npu_buffer_type> _default_buffer_type;
    uint32_t                         _dsp_domain_id = 0;

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
