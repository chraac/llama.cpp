
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ggml.h"

#include "ggml-backend.h"
#include "ggml-qnn.h"

#include "graph.hpp"
#include "qnn-lib.hpp"

namespace qnn {
typedef std::unordered_map<std::string, std::unique_ptr<qnn::ggml_qnn_graph>> ggml_qnn_graph_cache_t;
} // namespace qnn

struct ggml_backend_qnn_device_context {
    // initialize in constructor
    QNNBackend device;
    size_t threads;
    std::string name;
    std::string lib_name;

    // initialize in init
    qnn::qcom_socinfo socinfo = {};
    uint64_t supported_types;
    std::shared_ptr<qnn::qnn_instance> instance;
    std::shared_ptr<qnn::qnn_interface> qnn_interface;

    qnn::ggml_qnn_graph_cache_t qnn_graph_cache;

    explicit ggml_backend_qnn_device_context(QNNBackend device, size_t threads, const char *name,
                                             const char *lib_name) :
        device(device), threads(threads), name(name), lib_name(lib_name) {}
};
