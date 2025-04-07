#include "device.hpp"

namespace {

constexpr const ggml_guid kBackendNpuGuid = { 0x7a, 0xd7, 0x59, 0x7d, 0x8f, 0x66, 0x4f, 0x35,
                                              0x84, 0x8e, 0xf5, 0x9a, 0x9b, 0x83, 0x7d, 0x0a };

hexagon::npu_backend * get_backend_object(ggml_backend_t backend) {
    return reinterpret_cast<hexagon::npu_backend *>(backend);
}

const char * backend_get_name(ggml_backend_t backend) {
    auto * obj = get_backend_object(backend);
    return obj->get_name();
}

void backend_free(ggml_backend_t backend) {
    delete get_backend_object(backend);
}

bool backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src,
                              struct ggml_tensor * dst) {
    // TODO: implement this
    return false;
}

ggml_status backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // TODO: implement this
    return GGML_STATUS_SUCCESS;
}

}  // namespace

namespace hexagon {

npu_device::~npu_device() {
    if (_device_handle) {
        npu_device_close(_device_handle);
    }
}

bool npu_device::is_device_valid() const {
    if (!_rpc_mem) {
        LOG_ERROR("rpc memory not initialized\n");
        return false;
    }

    if (!_device_handle) {
        LOG_ERROR("NPU device not opened\n");
        return false;
    }

    return true;
}

bool npu_device::init_device(ggml_backend_dev_t dev, const char * params) {
    if (!_rpc_mem) {
        auto rpc_mem = std::make_shared<common::rpc_mem>();
        if (!rpc_mem->is_valid()) {
            LOG_ERROR("Failed to create rpc memory\n");
            return false;
        }

        _rpc_mem = std::move(rpc_mem);
    } else {
        LOG_DEBUG("NPU device is already initialized\n");
    }

    if (!_device_handle) {
        // TODO: fix uri here for each npu
        auto ret = npu_device_open(npu_device_URI, &_device_handle);
        if (ret != 0) {
            LOG_ERROR("ERROR 0x%x: Unable to open NPU device on domain %s\n", ret, npu_device_URI);
            _device_handle = 0;
            return false;
        }
    } else {
        LOG_DEBUG("NPU device is already opened\n");
    }

    _default_buffer_type = std::make_unique<hexagon::npu_buffer_type>(dev, _name + "_buffer_type", _rpc_mem);
    return true;
}

ggml_backend_buffer_type_t npu_device::get_default_buffer_type() {
    if (!_default_buffer_type) {
        LOG_ERROR("Default buffer type not initialized\n");
        return nullptr;
    }

    return _default_buffer_type.get();
}

npu_backend::npu_backend(npu_device * device) : ggml_backend{}, _device(device) {
    memccpy(&_guid, &kBackendNpuGuid, 0, sizeof(ggml_guid));
    guid                   = &_guid;
    iface.get_name         = backend_get_name;
    iface.free             = backend_free;
    iface.cpy_tensor_async = backend_cpy_tensor_async;
    iface.graph_compute    = backend_graph_compute;
}

}  // namespace hexagon
