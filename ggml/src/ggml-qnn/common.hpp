#pragma once

#include <memory>

#include "ggml-backend-impl.h"

enum backend_index_type {
    QNN_BACKEND_CPU = 0,
    QNN_BACKEND_GPU,
    QNN_BACKEND_NPU,

    HEXAGON_BACKEND,

    TOTAL_BACKEND_COUNT,
    QNN_BACKEND_COUNT = HEXAGON_BACKEND,
};

class backend_device_proxy {
  public:
    virtual ~backend_device_proxy() = default;

    virtual const ggml_backend_device_i & get_iface() const = 0;
    virtual void *                        get_context()     = 0;
};

using backend_device_proxy_ptr = std::shared_ptr<backend_device_proxy>;

backend_device_proxy_ptr create_qnn_backend_context(backend_index_type device);
backend_device_proxy_ptr create_hexagon_backend_context(backend_index_type device);

#define DISABLE_COPY(class_name)                 \
    class_name(const class_name &)     = delete; \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)            \
    class_name(class_name &&)     = delete; \
    void operator=(class_name &&) = delete
