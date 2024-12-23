
#include "graph.hpp"

#include <unordered_map>
#include <unordered_set>

#include "ggml-impl.h"

#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"

namespace {
using ggml_tensor_set_t = std::unordered_set<ggml_tensor *>;
using qnn_tensor_cache_t = std::unordered_map<ggml_tensor *, qnn::qnn_tensor_ptr_t>;

qnn::qnn_tensor_ptr_t create_tensor_with_cache(ggml_tensor *tensor, qnn::ggml_qnn_tensor::tensor_type_t type, int rank,
                                               QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                               qnn_tensor_cache_t &tensor_cache) {
    GGML_ASSERT(tensor);
    if (tensor_cache.count(tensor)) {
        return tensor_cache[tensor];
    }

    auto qnn_tensor = std::make_shared<qnn::ggml_qnn_tensor>(type, tensor->name, tensor->ne, tensor->type, rank, device,
                                                             graph_handle, qnn_instance);
    tensor_cache[tensor] = qnn_tensor;
    return qnn_tensor;
}

qnn::qnn_tensor_array_t create_tensors(const ggml_tensor_set_t &tensor_set, qnn::ggml_qnn_tensor::tensor_type_t type,
                                       int rank, QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                       std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                       qnn_tensor_cache_t &tensor_cache) {
    qnn::qnn_tensor_array_t tensors;
    for (auto *tensor : tensor_set) {
        tensors.push_back(
            create_tensor_with_cache(tensor, type, rank, device, graph_handle, qnn_instance, tensor_cache));
    }

    return tensors;
}

} // namespace

namespace qnn {

qnn_graph::qnn_graph(const std::string &graph_name, QNNBackend device, std::shared_ptr<qnn_instance> qnn_instance,
                     size_t vtcm_size_in_mb)
    : _graph_name(graph_name), _device(device), _qnn_instance(qnn_instance) {
    QNN_LOG_DEBUG("[%s][%s]created", get_backend_name(device), graph_name.c_str());

    auto qnn_interface = qnn_instance->get_qnn_interface();
    auto qnn_context = qnn_instance->get_qnn_context_handle();
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    Qnn_GraphHandle_t graph_handle = nullptr;
    if (device == QNN_BACKEND_NPU) {
        // TODO: fix graph config here for NPU
        QnnHtpGraph_CustomConfig_t hvx_config;
        hvx_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
        hvx_config.numHvxThreads = 8;
        QnnGraph_Config_t graph_hvx_config;
        graph_hvx_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_hvx_config.customConfig = &hvx_config;

        QnnHtpGraph_CustomConfig_t dlbc_config;
        dlbc_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        dlbc_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
        dlbc_config.optimizationOption.floatValue = 1.0; // set to 0.0 to turn off DLBC
        QnnGraph_Config_t graph_dlbc_config;
        graph_dlbc_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_dlbc_config.customConfig = &dlbc_config;

        QnnHtpGraph_CustomConfig_t opt_config;
        opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        opt_config.optimizationOption.floatValue = 1; // 1 / 3
        QnnGraph_Config_t graph_opt_config;
        graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_opt_config.customConfig = &opt_config;

        QnnHtpGraph_CustomConfig_t vtcm_config;
        vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcm_config.vtcmSizeInMB = vtcm_size_in_mb;
        QnnGraph_Config_t graph_vtcm_config;
        graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_vtcm_config.customConfig = &vtcm_config;

        const QnnGraph_Config_t *graph_configs[] = {&graph_hvx_config, &graph_dlbc_config, &graph_vtcm_config,
                                                    &graph_opt_config, nullptr};
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), graph_configs, &graph_handle);
    } else {
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), nullptr, &graph_handle);
    }

    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s", get_backend_name(device), graph_name.c_str(),
                      get_qnn_error_string(error));
        return;
    }

    QNN_LOG_INFO("[%s][%s]create succeed", get_backend_name(device), graph_name.c_str());
    _graph_handle = graph_handle;
    _qnn_interface = qnn_interface;
}

qnn_graph::~qnn_graph() { QNN_LOG_DEBUG("[%s][%s]destroy", get_backend_name(_device), _graph_name.c_str()); }

bool qnn_graph::build_graph(ggml_op_constructor_t op_constructor, const ggml_tensor_array_t &tensor_inputs,
                            const ggml_tensor_array_t &tensor_outputs) {
    GGML_ASSERT(op_constructor);
    if (!is_valid()) {
        QNN_LOG_ERROR("Invalid graph");
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]build_graph start", get_backend_name(_device), _graph_name.c_str());
    auto operation = op_constructor(_graph_name, _qnn_instance);
    if (!operation->initialize_op_nodes(_device, _graph_handle, tensor_inputs, tensor_outputs)) {
        QNN_LOG_ERROR("[%s][%s]initialize_op_nodes failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    _tensor_inputs = operation->get_input_tensors();
    _tensor_outputs = operation->get_output_tensors();
    _operations.push_back(std::move(operation));
    if (!qnn::add_op_to_graph(_graph_handle, _operations)) {
        QNN_LOG_ERROR("[%s]add nodes failed", _graph_name.c_str());
        return false;
    }

    auto error = _qnn_interface->qnn_graph_finalize(_graph_handle, nullptr, nullptr);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]qnn_graph_finalize.error: %s", get_backend_name(_device), _graph_name.c_str(),
                      get_qnn_error_string(error));
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]build_graph succeed", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::build_graph(qnn_op_config_array_t &operations, qnn_tensor_array_t &intputs,
                            qnn_tensor_array_t &outputs) {
    _tensor_inputs = intputs;
    _tensor_outputs = outputs;
    _operations = operations;
    if (!qnn::add_op_to_graph(_graph_handle, _operations)) {
        QNN_LOG_ERROR("[%s]add nodes failed", _graph_name.c_str());
        return false;
    }

    auto error = _qnn_interface->qnn_graph_finalize(_graph_handle, nullptr, nullptr);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]qnn_graph_finalize.error: %s", get_backend_name(_device), _graph_name.c_str(),
                      get_qnn_error_string(error));
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]build_graph succeed", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::execute(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) {
    if (!qnn::bind_tensors(tensor_inputs, _tensor_inputs, _qnn_tensor_inputs)) {
        QNN_LOG_ERROR("[%s][%s]bind input tensors failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    if (!qnn::bind_tensors(tensor_outputs, _tensor_outputs, _qnn_tensor_outputs)) {
        QNN_LOG_ERROR("[%s][%s]bind output tensors failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    auto &qnn_tensor_inputs = _qnn_tensor_inputs;
    auto &qnn_tensor_outputs = _qnn_tensor_outputs;
    auto error =
        _qnn_interface->qnn_graph_execute(_graph_handle, qnn_tensor_inputs.data(), qnn_tensor_inputs.size(),
                                          qnn_tensor_outputs.data(), qnn_tensor_outputs.size(), nullptr, nullptr);
    unbind_tensors(_tensor_inputs);
    unbind_tensors(_tensor_outputs);

    if (error != QNN_SUCCESS) {
        if (_device == QNN_BACKEND_NPU && error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION) {
            QNN_LOG_WARN("[%s][%s]NPU crashed. SSR detected. Caused QNN graph execute error.",
                         get_backend_name(_device), _graph_name.c_str());
        } else {
            QNN_LOG_ERROR("[%s][%s]error: %s", get_backend_name(_device), _graph_name.c_str(),
                          get_qnn_error_string(error));
        }
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]execute succeed", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool init_from_ggml_graph(const ggml_cgraph *cgraph, qnn_graph_ptr_t graph) {
    ggml_tensor_set_t input_set;
    ggml_tensor_set_t output_set;
    ggml_tensor_set_t visited_set;
    int rank = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *dst = cgraph->nodes[i];
        if (ggml_is_empty(dst)) {
            continue;
        }

        rank = std::max(rank, ggml_n_dims(dst));
        input_set.erase(dst);
        if (!visited_set.count(dst)) {
            output_set.insert(dst);
            visited_set.insert(dst);
        }

        for (size_t i = 0; i < GGML_MAX_DIMS && dst->src[i]; ++i) {
            auto *src = dst->src[i];
            rank = std::max(rank, ggml_n_dims(src));
            output_set.erase(src);
            if (!visited_set.count(src)) {
                input_set.insert(src);
                visited_set.insert(src);
            }
        }
    }

    qnn_tensor_cache_t tensor_cache;
    auto intput_tensors = create_tensors(input_set, ggml_qnn_tensor::INPUT, rank, graph->get_device(),
                                         graph->get_graph_handler(), graph->get_qnn_instance(), tensor_cache);
    auto output_tensors = create_tensors(output_set, ggml_qnn_tensor::OUTPUT, rank, graph->get_device(),
                                         graph->get_graph_handler(), graph->get_qnn_instance(), tensor_cache);
    qnn_op_config_array_t operations;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *dst = cgraph->nodes[i];
        if (ggml_is_empty(dst)) {
            continue;
        }

        QNN_LOG_DEBUG("[%s]create op: %s", get_backend_name(graph->get_device()), get_qnn_op_name(dst->op));
        auto qnn_op = create_op_constructor(dst->op);
        auto operation = qnn_op(dst->name, graph->get_qnn_instance()); // TODO: fix the name here

        // input tensors
        qnn_tensor_array_t input_qnn_tensors;
        for (size_t i = 0; i < get_qnn_op_input_param_count(dst->op); ++i) {
            auto input_qnn_tensor =
                create_tensor_with_cache(dst->src[i], ggml_qnn_tensor::INTERMEDIATE, rank, graph->get_device(),
                                         graph->get_graph_handler(), graph->get_qnn_instance(), tensor_cache);
            input_qnn_tensors.push_back(input_qnn_tensor);
        }
        operation->set_input_tensors(input_qnn_tensors);

        // output tensor
        qnn_tensor_array_t output_qnn_tensors =
            create_tensors({dst}, ggml_qnn_tensor::INTERMEDIATE, rank, graph->get_device(), graph->get_graph_handler(),
                           graph->get_qnn_instance(), tensor_cache);
        operation->set_output_tensors(output_qnn_tensors);

        operations.push_back(operation);
    }

    if (!graph->build_graph(operations, intput_tensors, output_tensors)) {
        QNN_LOG_ERROR("[%s]build graph failed", get_backend_name(graph->get_device()));
        return false;
    }

    return true;
}

} // namespace qnn
