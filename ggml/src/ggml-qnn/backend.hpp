
#pragma once

#include <unordered_map>

#include "ggml.h"

#include "ggml-backend.h"

#include "qnn.hpp"

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn::qnn_instance *instance;
    ggml_backend *backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
    qnn::qcom_socinfo socinfo;
    std::unordered_map<std::string, std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *, Qnn_Tensor_t *>> qnn_graph_map;
};