#include "device.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#include <domain_default.h>
#pragma GCC diagnostic pop

#include <remote.h>

#include "graph.hpp"
#include "util.hpp"

#define SKEL_URI_DEFINE(arch) ("file:///libhexagon_npu_skel_" arch ".so?npu_device_skel_handle_invoke&_modver=1.0")

namespace {

struct device_library_info {
    hexagon::hexagon_dsp_arch arch;
    const char *              device_lib_uri;
};

constexpr const device_library_info kDeviceLibraryInfo[] = {
    { hexagon::NONE, SKEL_URI_DEFINE("")    },
    { hexagon::V68,  SKEL_URI_DEFINE("v68") },
    { hexagon::V69,  SKEL_URI_DEFINE("v69") },
    { hexagon::V73,  SKEL_URI_DEFINE("v73") },
    { hexagon::V75,  SKEL_URI_DEFINE("v75") },
    { hexagon::V79,  SKEL_URI_DEFINE("v79") },
};

const device_library_info & get_device_library_info(hexagon::hexagon_dsp_arch arch) {
    for (const auto & info : kDeviceLibraryInfo) {
        if (info.arch == arch) {
            return info;
        }
    }

    LOG_ERROR("Unknown DSP arch: %d, using hexagon::NONE\n", arch);
    return kDeviceLibraryInfo[0];
}

const char * get_domain_param(uint32_t domain_id) {
    for (const auto & domain : supported_domains) {
        if ((uint32_t) domain.id == domain_id) {
            return domain.uri;
        }
    }

    return "";
}

constexpr const ggml_guid kBackendNpuGuid = { 0x7a, 0xd7, 0x59, 0x7d, 0x8f, 0x66, 0x4f, 0x35,
                                              0x84, 0x8e, 0xf5, 0x9a, 0x9b, 0x83, 0x7d, 0x0a };

hexagon::npu_backend * get_backend_object(ggml_backend_t backend) {
    return reinterpret_cast<hexagon::npu_backend *>(backend);
}

const char * backend_get_name(ggml_backend_t backend) {
    return get_backend_object(backend)->get_name();
}

void backend_free(ggml_backend_t backend) {
    delete get_backend_object(backend);
}

bool backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src,
                              ggml_tensor * dst) {
    // TODO: implement this
    return false;
}

ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return get_backend_object(backend)->graph_compute(cgraph);
}

}  // namespace

namespace hexagon {

// TODO: should we use another domain?
npu_device::npu_device(backend_index_type device) : _dsp_domain_id(CDSP_DOMAIN_ID) {
    GGML_UNUSED(device);
    LOG_DEBUG("[%s]NPU device created\n", _name.c_str());
}

npu_device::~npu_device() {
    if (_device_handle) {
        npu_device_close(_device_handle);
    }
}

size_t npu_device::get_alignment() const {
    uint32_t alignment = 0;
    npu_device_device_get_alignment(_device_handle, &alignment);
    return alignment;
}

bool npu_device::is_device_valid() const {
    if (!_rpc_mem) {
        LOG_ERROR("[%s]rpc memory not initialized\n", get_name());
        return false;
    }

    if (!_device_handle) {
        LOG_ERROR("[%s]NPU device not opened\n", get_name());
        return false;
    }

    return true;
}

bool npu_device::init_device(ggml_backend_dev_t dev, const char * params) {
    if (!_rpc_mem) {
        auto rpc_interface = std::make_shared<common::rpc_interface>();
        if (!rpc_interface->is_valid()) {
            LOG_ERROR("[%s]Failed to load rpc memory library\n", get_name());
            return false;
        }

        auto rpc_mem   = std::make_shared<common::rpc_mem>(rpc_interface);
        _rpc_interface = rpc_interface;
        _rpc_mem       = rpc_mem;
    } else {
        LOG_DEBUG("[%s]NPU device is already initialized\n", get_name());
    }

    if (!_device_handle) {
        auto         arch            = get_dsp_arch(_rpc_interface, _dsp_domain_id);
        const auto & device_lib_info = get_device_library_info(arch);
        std::string  device_lib_uri  = device_lib_info.device_lib_uri;
        device_lib_uri += get_domain_param(_dsp_domain_id);
        LOG_DEBUG("[%s]NPU device arch: %d, uri: %s\n", get_name(), arch, device_lib_uri.c_str());
        auto err = npu_device_open(device_lib_uri.c_str(), &_device_handle);
        if (err != AEE_SUCCESS) {
            if (err == AEE_ECONNREFUSED) {
                LOG_DEBUG("[%s]NPU device is not available, trying to enable unsigned DSP module and reopen\n",
                          get_name());
                enable_unsigned_dsp_module(_rpc_interface, _dsp_domain_id);
                err = npu_device_open(device_lib_uri.c_str(), &_device_handle);
            }

            if (err != AEE_SUCCESS) {
                LOG_ERROR("[%s]Unable to open NPU device, err: 0x%x, uri %s\n", get_name(), err,
                          device_lib_uri.c_str());
                _device_handle = 0;
                return false;
            }
        }

        LOG_DEBUG("[%s]NPU device opened successfully\n", get_name());
    } else {
        LOG_DEBUG("[%s]NPU device is already opened\n", get_name());
    }

    _default_buffer_type = std::make_unique<hexagon::host_buffer_type>(dev, _name + "_buffer_type", _rpc_mem);
    return true;
}

bool npu_device::supports_buft(ggml_backend_buffer_type_t buft) const {
    return buft && buft->device && buft->device->context == this;
}

bool npu_device::supports_op_impl(const ggml_tensor * op) {
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (type_to_npu_type(op->type) == NPU_DATA_TYPE_COUNT) {
        LOG_DEBUG("[%s]Unsupported op tensor type: %s\n", get_name(), ggml_type_name(op->type));
        return false;
    }

    auto * src0 = op->src[0];
    if (!src0) {
        LOG_DEBUG("[%s]Unsupported inplace op: %s\n", get_name(), ggml_op_name(op->op));
        return false;
    }

    if (type_to_npu_type(src0->type) == NPU_DATA_TYPE_COUNT) {
        LOG_DEBUG("[%s]Unsupported src0 tensor type: %s\n", get_name(), ggml_type_name(src0->type));
        return false;
    }

    auto * src1 = op->src[1];
    if (src1 && type_to_npu_type(src1->type) == NPU_DATA_TYPE_COUNT) {
        LOG_DEBUG("[%s]Unsupported src1 tensor type: %s\n", get_name(), ggml_type_name(src1->type));
        return false;
    }

    if (op_to_npu_op(op->op) == NPU_OP_COUNT) {
        LOG_DEBUG("[%s]Unsupported op: %s\n", get_name(), ggml_op_name(op->op));
        return false;
    }

    constexpr const auto get_spec = [](const ggml_tensor * tensor) -> npu_device_tensor_spec {
        if (!tensor) {
            return npu_device_tensor_spec{};
        }

        static_assert(DEVICE_TENSOR_MAX_DIMS == GGML_MAX_DIMS, "tensor dimensions mismatch");
        npu_device_tensor_spec spec{};
        spec.ne[0] = tensor->ne[0];
        spec.ne[1] = tensor->ne[1];
        spec.ne[2] = tensor->ne[2];
        spec.ne[3] = tensor->ne[3];
        spec.type  = type_to_npu_type(tensor->type);
        return spec;
    };

    boolean supported = false;
    auto    src0_spec = get_spec(src0);
    auto    src1_spec = get_spec(src1);
    auto    dst_spec  = get_spec(op);
    auto    ret = npu_device_device_support_op(_device_handle, &src0_spec, &src1_spec, &dst_spec, op_to_npu_op(op->op),
                                               &supported);
    if (ret != AEE_SUCCESS || !supported) {
        LOG_DEBUG("[%s]Unsupported op: %s, ret: 0x%x, supported: %d\n", get_name(), ggml_op_name(op->op), ret,
                  supported);
        return false;
    }

    LOG_DEBUG("[%s]Supported op: %s\n", get_name(), ggml_op_name(op->op));
    return true;
}

bool npu_device::offload_op(const ggml_tensor * op) {
    // TODO: implement this
    return false;
}

ggml_backend_buffer_type_t npu_device::get_default_buffer_type() {
    if (!_default_buffer_type) {
        LOG_ERROR("[%s]Default buffer type not initialized\n", get_name());
        return nullptr;
    }

    return _default_buffer_type.get();
}

npu_backend::npu_backend(ggml_backend_dev_t dev) : ggml_backend{} {
    memccpy(&_guid, &kBackendNpuGuid, 0, sizeof(ggml_guid));
    device                 = dev;
    guid                   = &_guid;
    iface.get_name         = backend_get_name;
    iface.free             = backend_free;
    iface.cpy_tensor_async = backend_cpy_tensor_async;
    iface.graph_compute    = backend_graph_compute;
    _device                = reinterpret_cast<npu_device *>(dev->context);
}

ggml_status npu_backend::graph_compute(ggml_cgraph * cgraph) {
    if (!cgraph || !cgraph->n_nodes) {
        LOG_DEBUG("[%s]Graph is empty, nothing to compute\n", get_name());
        return GGML_STATUS_SUCCESS;
    }

    std::shared_ptr<host_graph> graph;
    if (_graph_cache.count(cgraph) == 0) {
        LOG_DEBUG("[%s]graph(%p) not found in cache, creating new graph\n", get_name(), (void *) cgraph);
        graph = std::make_shared<host_graph>(cgraph, _device->get_device_handle());
        if (!graph->is_valid()) {
            LOG_ERROR("Failed to create graph\n");
            return GGML_STATUS_FAILED;
        }

        _graph_cache[cgraph] = graph;
    } else {
        graph = _graph_cache[cgraph];
        LOG_DEBUG("[%s]graph(%p) found in cache, using existing graph\n", get_name(), (void *) cgraph);
        if (!graph->update(cgraph)) {
            LOG_ERROR("[%s]Failed to update graph(%p)\n", get_name(), (void *) cgraph);
            return GGML_STATUS_FAILED;
        }
    }

    return graph->compute() ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

}  // namespace hexagon
