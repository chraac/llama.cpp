#include "ggml-qnn.h"

#include <unistd.h>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-qnn/backend-ops.hpp"
#include "ggml-qnn/backend.hpp"
#include "ggml-qnn/logger.hpp"
#include "ggml-qnn/tensor.hpp"
#include "ggml-qnn/utils.hpp"

// =================================================================================================
//
//  self-defined macro / data structure
//
// =================================================================================================
#ifdef NDEBUG
#define ENABLE_QNNBACKEND_PERF 0 // enable/disable op's perf info
#else
#define ENABLE_QNNBACKEND_PERF 1 // enable/disable op's perf info
#endif

#define QNN_BACKEND_NAME "qnn"

namespace {

struct qnn_device_caps {
    const char *name;
    const char *description;
    const char *lib_name;
    enum ggml_backend_dev_type type;

    // TODO: should get this caps from device
    std::unordered_set<ggml_type> supported_types;
};

const qnn_device_caps kDeviceCaps[GGML_QNN_MAX_DEVICES] = {
    {// https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/CpuOpDefSupplement.html#matmul
     "qnn-cpu",
     "Qualcomm Kryo CPU",
     "libQnnCpu.so",
     GGML_BACKEND_DEVICE_TYPE_CPU,
     {GGML_TYPE_F32, GGML_TYPE_I8}},
    {// https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/GpuOpDefSupplement.html#matmul
     "qnn-gpu",
     "Qualcomm Adreno GPU",
     "libQnnGpu.so",
     GGML_BACKEND_DEVICE_TYPE_GPU,
     {GGML_TYPE_F32, GGML_TYPE_F16}},
    {// https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/HtpOpDefSupplement.html#matmul
     "qnn-npu",
     "Qualcomm NPU",
     "libQnnHtp.so",
     GGML_BACKEND_DEVICE_TYPE_GPU,
     {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I16, GGML_TYPE_I8}},
};

class ggml_backend_qnn_buffer_context {
public:
    ggml_backend_qnn_buffer_context(QNNBackend device, std::shared_ptr<qnn::qnn_instance> instance, size_t size)
        : _instance(instance), _name(QNN_BACKEND_NAME + std::to_string(device)) {

        // TODO: fix this for other platforms
        size_t size_page = sysconf(_SC_PAGESIZE);

        // TODO: for qnn npu, a better way here is to reuse the buffer allocated by qnn rpc, will save an extra copy
        _buffer = qnn::align_alloc(size_page, size);

        if (!_buffer) {
            QNN_LOG_WARN("failed to allocate %.2f MiB\n", float(size / (1 << 20)));
            return;
        }

        _buffer_size = size;
    }

    ~ggml_backend_qnn_buffer_context() {
        // the free will do nothing if the _buffer is nullptr
        qnn::align_free(_buffer);
    }

    bool is_valid() const { return _buffer != nullptr; }

    void *get_buffer() { return _buffer; }
    size_t get_buffer_size() { return _buffer_size; }

private:
    std::shared_ptr<qnn::qnn_instance> _instance;
    std::string _name;
    void *_buffer = nullptr;
    size_t _buffer_size = 0;
};

struct ggml_backend_qnn_buffer_type_context {
    std::string name;
};

ggml_backend_qnn_device_context *get_device_context(ggml_backend_dev_t dev) {
    return reinterpret_cast<ggml_backend_qnn_device_context *>(dev->context);
}

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend buffer object
 * -----------------------------------------------------------------------------------------------
 */
void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    delete ctx;
}

void *ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    return ctx->get_buffer();
}

void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
    // Do nothing here, the qnn tensor will be create along with the graph.
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
}

void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor, const void *data,
                                        size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}

void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor *tensor, void *data,
                                        size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}

bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor *src,
                                        struct ggml_tensor *dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    memset(ctx->get_buffer(), value, ctx->get_buffer_size());
}

ggml_backend_buffer_i ggml_backend_qnn_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_qnn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_qnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_qnn_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_qnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_qnn_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_qnn_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_qnn_buffer_clear,
    /* .reset           = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend object
 * -----------------------------------------------------------------------------------------------
 */
const char *ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_QNN_NAME;
}

ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto *dev_ctx = get_device_context(buft->device);
    ggml_backend_qnn_buffer_context *ctx =
        new ggml_backend_qnn_buffer_context((QNNBackend)dev_ctx->device, dev_ctx->instance, size);
    if (!ctx->is_valid()) {
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}

size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

// TODO: this value is an experimental value, works fine with whisper/llm/minicpm-v inference on Android
size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (96 * 1024 * 1024);
}

bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    // TODO: fix this
    GGML_UNUSED(buft);
    return true;
}

const char *ggml_backend_qnn_name(ggml_backend_t backend) {
    auto *device_ctx = get_device_context(backend->device);
    return device_ctx->name.c_str();
}

void ggml_backend_qnn_free(ggml_backend_t backend) {
    auto *device_ctx = get_device_context(backend->device);
    QNN_LOG_INFO("idx %d, name:%s", device_ctx->device, device_ctx->name.c_str());

    auto &instance = device_ctx->instance;
    if (instance) {
        device_ctx->qnn_graph_cache.clear();
        device_ctx->qnn_interface.reset();
        instance->qnn_finalize();
        instance.reset();
    }
}

ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_qnn_buffer_type_context ggml_backend_qnn_buffer_type_contexts[GGML_QNN_MAX_DEVICES];
    static ggml_backend_buffer_type ggml_backend_qnn_buffer_types[GGML_QNN_MAX_DEVICES];
    static bool ggml_backend_qnn_buffer_type_initialized = false;
    auto *dev_ctx = get_device_context(dev);
    if (!ggml_backend_qnn_buffer_type_initialized) {
        for (size_t i = 0; i < GGML_QNN_MAX_DEVICES; i++) {
            auto &context = ggml_backend_qnn_buffer_type_contexts[i];
            context = {std::string(QNN_BACKEND_NAME) + std::to_string(i)};
            ggml_backend_qnn_buffer_types[i] = {
                /* .iface   = */ {
                    /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                    /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                    /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                    /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
                    /* .is_host          = */ ggml_backend_qnn_buffer_is_host,
                },
                /* .device */ dev,
                /* .context = */ &context,
            };
        }
        ggml_backend_qnn_buffer_type_initialized = true;
    }

    return &ggml_backend_qnn_buffer_types[dev_ctx->device];
}

ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph *cgraph) {
    enum ggml_status result = GGML_STATUS_SUCCESS;
    auto *device_ctx = get_device_context(backend->device);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
            node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = qnn::ggml_qnn_forward(device_ctx, node);
        if (!ok) {
            QNN_LOG_DEBUG("error: op not supported %s (%s)\n", node->name, ggml_op_name(node->op));
        }
    }

    return result;
}

ggml_backend_i ggml_backend_qnn_interface = {
    /* .get_name                = */ ggml_backend_qnn_name,
    /* .free                    = */ ggml_backend_qnn_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend device object
 * -----------------------------------------------------------------------------------------------
 */
const char *ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev) {
    const auto &caps = kDeviceCaps[get_device_context(dev)->device];
    return caps.name;
}

const char *ggml_backend_qnn_device_get_description(ggml_backend_dev_t dev) {
    const auto &caps = kDeviceCaps[get_device_context(dev)->device];
    return caps.description;
}

void ggml_backend_qnn_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    // TODO: get memory info
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

enum ggml_backend_dev_type ggml_backend_qnn_device_get_type(ggml_backend_dev_t dev) {
    switch (get_device_context(dev)->device) {
        case QNN_BACKEND_CPU:
            return GGML_BACKEND_DEVICE_TYPE_CPU;
        case QNN_BACKEND_GPU:
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        case QNN_BACKEND_NPU:
            return GGML_BACKEND_DEVICE_TYPE_ACCEL;
        default:
            break;
    }
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

void ggml_backend_qnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    props->name = ggml_backend_qnn_device_get_name(dev);
    props->description = ggml_backend_qnn_device_get_description(dev);
    props->type = ggml_backend_qnn_device_get_type(dev);
    ggml_backend_qnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* async                */ false,
        /* host_buffer          */ false,
        /* buffer_from_host_ptr */ false,
        /* events               */ false,
    };
}

ggml_guid_t ggml_backend_qnn_guid() {
    static ggml_guid guid = {0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
                             0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09};
    return &guid;
}

ggml_backend_t ggml_backend_qnn_init_with_device_context(ggml_backend_dev_t dev, const char *extend_lib_search_path) {
    if (!extend_lib_search_path) {
        extend_lib_search_path = GGML_QNN_DEFAULT_LIB_SEARCH_PATH;
        QNN_LOG_WARN("extend_lib_search_path is nullptr, will use " GGML_QNN_DEFAULT_LIB_SEARCH_PATH " as default");
    }

    auto *dev_ctx = get_device_context(dev);
    const auto device = dev_ctx->device;
    QNN_LOG_DEBUG("device %d", device);
    QNN_LOG_DEBUG("extend_lib_search_path %s", extend_lib_search_path);
    std::string path = extend_lib_search_path;

// TODO: Fix this for other platforms
#if defined(__ANDROID__) || defined(ANDROID)
    if (device == QNN_BACKEND_NPU) {
        if (setenv("LD_LIBRARY_PATH",
                   (path + ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/"
                           "dsp:/vendor/dsp/images")
                       .c_str(),
                   1) == 0) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
        if (setenv("ADSP_LIBRARY_PATH",
                   (path + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/"
                           "rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp")
                       .c_str(),
                   1) == 0) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
    } else {
        if (setenv("LD_LIBRARY_PATH", path.c_str(), 1) == 0) {
            QNN_LOG_INFO("%s backend setenv successfully\n", qnn::get_backend_name(device));
        } else {
            QNN_LOG_ERROR("%s backend setenv failure\n", qnn::get_backend_name(device));
        }
    }
#endif

    auto instance = std::make_shared<qnn::qnn_instance>(path, dev_ctx->lib_name, "ggml");
    auto result = instance->qnn_init(nullptr);
    if (result != 0) {
        QNN_LOG_WARN("init qnn subsystem failed with qnn backend %s, pls check why\n", qnn::get_backend_name(device));
        return nullptr;
    }
    auto qnn_interface = instance->get_qnn_interface();
    if (!qnn_interface) {
        QNN_LOG_WARN("qnn subsystem failure\n");
        return nullptr;
    }

    std::string device_name = qnn::get_backend_name(device);
    QNN_LOG_INFO("qnn device name %s", device_name.c_str());
    dev_ctx->instance = instance;
    dev_ctx->qnn_interface = qnn_interface;
    dev_ctx->socinfo = instance->get_soc_info();
    dev_ctx->supported_types = kDeviceCaps[device].supported_types;

    ggml_backend_t qnn_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_qnn_guid(),
        /* .iface     = */ ggml_backend_qnn_interface,
        /* .device    = */ dev,
        /* .context   = */ nullptr,
    };

    return qnn_backend;
}

ggml_backend_t ggml_backend_qnn_device_init(ggml_backend_dev_t dev, const char *params) {
    return ggml_backend_qnn_init_with_device_context(dev, params);
}

ggml_backend_buffer_type_t ggml_backend_qnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_qnn_buffer_type(dev);
}

ggml_backend_buffer_t ggml_backend_qnn_device_buffer_from_ptr(ggml_backend_dev_t dev, void *ptr, size_t size,
                                                              size_t max_tensor_size) {
    // TODO
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

bool ggml_backend_qnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor *op) {
    auto *device_ctx = get_device_context(dev);
    return qnn::ggml_qnn_supports_op(device_ctx, op);
}

bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

const struct ggml_backend_device_i ggml_backend_qnn_device_interface = {
    /* .get_name             = */ ggml_backend_qnn_device_get_name,
    /* .get_description      = */ ggml_backend_qnn_device_get_description,
    /* .get_memory           = */ ggml_backend_qnn_device_get_memory,
    /* .get_type             = */ ggml_backend_qnn_device_get_type,
    /* .get_props            = */ ggml_backend_qnn_device_get_props,
    /* .init_backend         = */ ggml_backend_qnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_qnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_qnn_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_qnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_qnn_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend registry object
 * -----------------------------------------------------------------------------------------------
 */

struct ggml_backend_qnn_reg_impl : ggml_backend_reg {
    std::array<std::unique_ptr<ggml_backend_qnn_device_context>, GGML_QNN_MAX_DEVICES> device_contexts;
    std::array<ggml_backend_device, GGML_QNN_MAX_DEVICES> devices;

    explicit ggml_backend_qnn_reg_impl(ggml_backend_reg_i interface) {
        context = this;
        iface = interface;

        for (int i = 0; i < GGML_QNN_MAX_DEVICES; i++) {
            const auto device_enum = (QNNBackend)(GGML_QNN_MAX_DEVICES - 1 - i); // init from the last device, i.e. NPU
            device_contexts[i] = std::make_unique<ggml_backend_qnn_device_context>(
                /* .device   = */ device_enum, // init from the last device, i.e. NPU
                /* .threads  = */ 1,
                /* .name     = */ qnn::get_backend_name(device_enum),
                /* .lib_name = */ kDeviceCaps[device_enum].lib_name);

            auto &device = devices[i];
            device.iface = ggml_backend_qnn_device_interface;
            device.reg = this;
            device.context = device_contexts[i].get();
        }
    }
};

const char *ggml_backend_qnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_QNN_NAME;
}

size_t ggml_backend_qnn_reg_get_device_count(ggml_backend_reg_t reg) {
    auto *ctx = (ggml_backend_qnn_reg_impl *)reg->context;
    return ctx->devices.size();
}

ggml_backend_dev_t ggml_backend_qnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto *ctx = (ggml_backend_qnn_reg_impl *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &(ctx->devices[index]);
}

const ggml_backend_reg_i ggml_backend_qnn_reg_interface = {
    /* .get_name         = */ ggml_backend_qnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_qnn_reg_get_device_count,
    /* .get_device_get   = */ ggml_backend_qnn_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

} // namespace

ggml_backend_reg_t ggml_backend_qnn_reg() {
    static ggml_backend_qnn_reg_impl reg{ggml_backend_qnn_reg_interface};
    return &reg;
}
