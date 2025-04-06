
#include <memory>
#include <string>

#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "rpc-mem.hpp"

namespace {

constexpr const ggml_guid kBackendNpuGuid = { 0x7a, 0xd7, 0x59, 0x7d, 0x8f, 0x66, 0x4f, 0x35,
                                              0x84, 0x8e, 0xf5, 0x9a, 0x9b, 0x83, 0x7d, 0x0a };

class npu_device_impl {
  public:
    static npu_device_impl * get_device(ggml_backend_dev_t device) {
        return reinterpret_cast<npu_device_impl *>(device->context);
    }

    static npu_device_impl * get_device(ggml_backend_t backend) { return npu_device_impl::get_device(backend->device); }

    static const char * get_name(ggml_backend_dev_t dev) {
        // TODO: implement this
        GGML_UNUSED(dev);
        return "hexagon-npu";
    }

    static const char * get_description(ggml_backend_dev_t dev) {
        // TODO: implement this
        GGML_UNUSED(dev);
        return "Hexagon NPU";
    }

    static void get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
        // TODO: implement this
        GGML_UNUSED(dev);
        *free  = 0;
        *total = 0;
    }

    static enum ggml_backend_dev_type get_type(ggml_backend_dev_t dev) {
        GGML_UNUSED(dev);
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }

    static void get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
        props->name        = npu_device_impl::get_name(dev);
        props->description = npu_device_impl::get_description(dev);
        props->type        = npu_device_impl::get_type(dev);
        npu_device_impl::get_memory(dev, &props->memory_free, &props->memory_total);
        props->caps = {};
    }

    static ggml_backend_t             init_backend(ggml_backend_dev_t dev, const char * params);
    static ggml_backend_buffer_type_t get_buffer_type(ggml_backend_dev_t dev);
    static ggml_backend_buffer_type_t get_host_buffer_type(ggml_backend_dev_t dev);
    static ggml_backend_buffer_t      buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size,
                                                           size_t max_tensor_size);
    static bool                       supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);

    static bool supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        if (!is_npu_device(dev)) {
            return false;
        }

        // TODO: check if the buffer type is for this device
        return true;
    }

    static bool offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);

    static bool is_npu_device(ggml_backend_dev_t dev) {
        return dev->iface.get_name(dev) == npu_device_impl::get_name(dev);
    }

    explicit npu_device_impl(backend_index_type device) : _device(device) {}

    bool init_device(const char * params) {
        auto rpc_mem = std::make_unique<common::rpc_mem>();
        if (!rpc_mem->is_valid()) {
            LOG_ERROR("Failed to create rpc memory\n");
            return false;
        }

        // TODO: load the NPU library and initialize it here

        _rpc_mem = std::move(rpc_mem);
        return true;
    }

  private:
    backend_index_type               _device;
    std::unique_ptr<common::rpc_mem> _rpc_mem;

    DISABLE_COPY(npu_device_impl);
    DISABLE_MOVE(npu_device_impl);
};

class npu_backend_impl : public ggml_backend {
  public:
    static npu_backend_impl * get_backend(ggml_backend_t backend) {
        return reinterpret_cast<npu_backend_impl *>(backend);
    }

    static const char * get_name(ggml_backend_t backend) {
        auto * impl = get_backend(backend);
        return impl->_name.c_str();
    }

    static void free(ggml_backend_t backend) { delete get_backend(backend); }

    static bool cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src,
                                 struct ggml_tensor * dst) {
        // TODO: implement this
    }

    static ggml_status graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
        // TODO: implement this
    }

    explicit npu_backend_impl(npu_device_impl * device_impl) : ggml_backend{}, _device_impl(device_impl) {
        memccpy(&_guid, &kBackendNpuGuid, 0, sizeof(ggml_guid));
        this->guid                   = &_guid;
        this->iface.get_name         = npu_backend_impl::get_name;
        this->iface.free             = npu_backend_impl::free;
        this->iface.cpy_tensor_async = npu_backend_impl::cpy_tensor_async;
        this->iface.graph_compute    = npu_backend_impl::graph_compute;
    }

    ~npu_backend_impl() {}

  private:
    ggml_guid         _guid        = {};
    std::string       _name        = npu_device_impl::get_name(nullptr);  // use device name by default
    npu_device_impl * _device_impl = nullptr;
};

ggml_backend_t npu_device_impl::init_backend(ggml_backend_dev_t dev, const char * params) {
    auto * device = npu_device_impl::get_device(dev);
    if (!device->init_device(params)) {
        LOG_ERROR("[%s]Failed to init device\n", npu_device_impl::get_name(dev));
        return nullptr;
    }

    return new npu_backend_impl(device);
}

constexpr const ggml_backend_device_i npu_device_interface = {
    /* .get_name             = */ npu_device_impl::get_name,
    /* .get_description      = */ npu_device_impl::get_description,
    /* .get_memory           = */ npu_device_impl::get_memory,
    /* .get_type             = */ npu_device_impl::get_type,
    /* .get_props            = */ npu_device_impl::get_props,
    /* .init_backend         = */ npu_device_impl::init_backend,
    /* .get_buffer_type      = */ npu_device_impl::get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ npu_device_impl::buffer_from_host_ptr,
    /* .supports_op          = */ npu_device_impl::supports_op,
    /* .supports_buft        = */ npu_device_impl::supports_buft,
    /* .offload_op           = */ npu_device_impl::offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

class npu_device_proxy : public backend_device_proxy {
  public:
    explicit npu_device_proxy(backend_index_type device) { _device_impl = std::make_unique<npu_device_impl>(device); }

    const ggml_backend_device_i & get_iface() const { return npu_device_interface; }

    void * get_context() { return _device_impl.get(); }

  private:
    std::unique_ptr<npu_device_impl> _device_impl;

    DISABLE_COPY(npu_device_proxy);
    DISABLE_MOVE(npu_device_proxy);
};

}  // namespace

backend_device_proxy_ptr create_hexagon_backend_context(backend_index_type device) {
    if (device < QNN_BACKEND_COUNT || device >= TOTAL_BACKEND_COUNT) {
        return backend_device_proxy_ptr();
    }

    // TODO: implement hexagon backend context creation
    return backend_device_proxy_ptr();
}
