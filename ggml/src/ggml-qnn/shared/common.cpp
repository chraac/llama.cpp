
#include "common.hpp"

#include <memory>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-qnn.h"

namespace {

struct ggml_backend_qnn_reg_impl : ggml_backend_reg {
    std::vector<backend_device_proxy_ptr> device_proxies;
    std::vector<ggml_backend_device>      devices;

    explicit ggml_backend_qnn_reg_impl(ggml_backend_reg_i interface) {
        context = this;
        iface   = interface;

        LOG_INFO("backend registry init\n");
        for (size_t i = 0; i < TOTAL_BACKEND_COUNT; i++) {
            const auto device_enum =
                (backend_index_type) (TOTAL_BACKEND_COUNT - 1 - i);  // init from the last device, i.e. NPU

            backend_device_proxy_ptr device_proxy;
            if (device_enum < QNN_BACKEND_COUNT) {
#ifdef GGML_HEXAGON_NPU_ONLY
                device_proxy = create_qnn_backend_context(device_enum);
#else
                LOG_DEBUG("skip qnn device %d\n", (int) device_enum);
#endif
            } else {
#ifdef GGML_QNN_ENABLE_HEXAGON_BACKEND
                device_proxy = create_hexagon_backend_context(device_enum);
#else
                LOG_DEBUG("skip hexagon device %d\n", (int) device_enum);
                continue;
#endif
            }

            if (!device_proxy) {
                LOG_DEBUG("skip device %d\n", (int) device_enum);
                continue;
            }

            devices.emplace_back(ggml_backend_device{
                /* iface = */ device_proxy->get_iface(),
                /* reg = */ this,
                /* context = */ device_proxy->get_context(),
            });

            device_proxies.emplace_back(device_proxy);
        }
    }
};

const char * ggml_backend_qnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "qnn";
}

size_t ggml_backend_qnn_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = (ggml_backend_qnn_reg_impl *) reg->context;
    return ctx->devices.size();
}

ggml_backend_dev_t ggml_backend_qnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_qnn_reg_impl *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &(ctx->devices[index]);
}

const ggml_backend_reg_i ggml_backend_qnn_reg_interface = {
    /* .get_name         = */ ggml_backend_qnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_qnn_reg_get_device_count,
    /* .get_device_get   = */ ggml_backend_qnn_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

}  // namespace

ggml_backend_reg_t ggml_backend_qnn_reg() {
    static ggml_backend_qnn_reg_impl reg{ ggml_backend_qnn_reg_interface };
    return &reg;
}
