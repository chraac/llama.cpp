
#include "graph.hpp"

#include <set>
#include <unordered_map>

#include "ggml-impl.h"

#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"

namespace {
using qnn_tensor_cache_t = std::unordered_map<ggml_tensor *, qnn::qnn_tensor_ptr_t>;

int get_op_max_rank(const ggml_tensor *op) {
    int max_rank = ggml_n_dims(op);
    const int count = (int)qnn::get_qnn_op_input_param_count(qnn::get_qnn_op_index(op));
    for (int i = 0; i < count; ++i) {
        max_rank = std::max(max_rank, ggml_n_dims(op->src[i]));
    }

    return max_rank;
}

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

qnn::qnn_tensor_array_t create_tensors_with_cache(const qnn::ggml_tensor_array_t &ggml_tensors,
                                                  qnn::ggml_qnn_tensor::tensor_type_t type, int rank, QNNBackend device,
                                                  Qnn_GraphHandle_t graph_handle,
                                                  std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                  qnn_tensor_cache_t &tensor_cache) {
    qnn::qnn_tensor_array_t tensors;
    for (auto *tensor : ggml_tensors) {
        tensors.push_back(
            create_tensor_with_cache(tensor, type, rank, device, graph_handle, qnn_instance, tensor_cache));
    }

    return tensors;
}

qnn::qnn_op_config_ptr_t create_operation_from_op_tensor(ggml_tensor *dst, const std::string &name, int rank,
                                                         QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                                         std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                         bool is_intermediate, qnn_tensor_cache_t &tensor_cache) {
    const auto op_index = qnn::get_qnn_op_index(dst);
    auto qnn_op = qnn::create_op_constructor(op_index);
    auto operation = qnn_op(name, qnn_instance);

    // input tensors
    qnn::qnn_tensor_array_t input_qnn_tensors;
    auto tensor_type = is_intermediate ? qnn::ggml_qnn_tensor::INTERMEDIATE : qnn::ggml_qnn_tensor::INPUT;
    for (size_t i = 0; i < qnn::get_qnn_op_input_param_count(op_index); ++i) {
        auto input_qnn_tensor =
            create_tensor_with_cache(dst->src[i], tensor_type, rank, device, graph_handle, qnn_instance, tensor_cache);
        input_qnn_tensors.push_back(input_qnn_tensor);
    }
    operation->set_input_tensors(input_qnn_tensors);

    // output tensor
    tensor_type = is_intermediate ? qnn::ggml_qnn_tensor::INTERMEDIATE : qnn::ggml_qnn_tensor::OUTPUT;
    qnn::qnn_tensor_array_t output_qnn_tensors =
        create_tensors_with_cache({dst}, tensor_type, rank, device, graph_handle, qnn_instance, tensor_cache);
    operation->set_output_tensors(output_qnn_tensors);

    // initialize operation
    if (!operation->initialize_op_nodes(device, graph_handle)) {
        QNN_LOG_ERROR("[%s][%s]initialize_op_nodes failed", qnn::get_backend_name(device), name.c_str());
        return nullptr;
    }

    return operation;
}

bool bind_src_tensors(ggml_tensor *op, qnn::qnn_tensor_array_t &tensor_wrappers,
                      std::vector<Qnn_Tensor_t> &qnn_tensors) {
    if (op->op == GGML_OP_NONE) {
        QNN_LOG_DEBUG("op %s is not a valid op", ggml_get_name(op));
        return false;
    }

    const auto param_count = qnn::get_qnn_op_input_param_count(qnn::get_qnn_op_index(op));
    GGML_ASSERT(tensor_wrappers.size() == param_count);
    qnn_tensors.resize(param_count);
    for (size_t i = 0; i < param_count; ++i) {
        auto *ggml_tensor = op->src[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor)) {
            QNN_LOG_ERROR("bind tensor %s failed", ggml_get_name(ggml_tensor));
            return false;
        }

        qnn_tensors[i] = tensor_wrappers[i]->get_qnn_tensor();
    }

    return true;
}

int get_io_tensors_from_graph(const ggml_cgraph *cgraph, qnn::ggml_tensor_array_t &inputs,
                              qnn::ggml_tensor_array_t &outputs) {
    using ggml_tensor_set_t = std::set<ggml_tensor *>;

    ggml_tensor_set_t input_set;
    ggml_tensor_set_t output_set;
    ggml_tensor_set_t visited_set;
    int rank = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *dst = cgraph->nodes[i];
        if (ggml_is_empty(dst)) {
            continue;
        }

        if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW) {
            // TODO: remove GGML_OP_VIEW after view op is supported
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

    inputs.assign(input_set.begin(), input_set.end());
    outputs.assign(output_set.begin(), output_set.end());
    return rank;
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

bool qnn_graph::build_graph_from_op(ggml_tensor *op) {
    if (!is_valid()) {
        QNN_LOG_ERROR("Invalid graph");
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]build start", get_backend_name(_device), _graph_name.c_str());
    qnn_tensor_cache_t tensor_cache;
    const auto rank = get_op_max_rank(op);
    auto operation = create_operation_from_op_tensor(op, _graph_name, rank, _device, _graph_handle, _qnn_instance,
                                                     false, tensor_cache);
    if (!operation) {
        QNN_LOG_ERROR("[%s][%s]create_operation_from_op_tensor failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    _tensor_inputs = operation->get_input_tensors();
    _tensor_outputs = operation->get_output_tensors();
    _operations.push_back(std::move(operation));
    if (!finalize()) {
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]build succeed", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::build_graph_from_ggml_graph(const ggml_cgraph *cgraph) {
    QNN_LOG_DEBUG("[%s][%s]build start", get_backend_name(_device), _graph_name.c_str());

    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
    int rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s]rank: %d, input_set: %d, output_set: %d", get_backend_name(_device), rank, int(inputs.size()),
                  int(outputs.size()));

    {
        qnn_tensor_cache_t tensor_cache;
        auto input_tensors = create_tensors_with_cache(inputs, ggml_qnn_tensor::INPUT, rank, _device, _graph_handle,
                                                       _qnn_instance, tensor_cache);
        auto output_tensors = create_tensors_with_cache(outputs, ggml_qnn_tensor::OUTPUT, rank, _device, _graph_handle,
                                                        _qnn_instance, tensor_cache);
        qnn_op_config_array_t operations;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor *dst = cgraph->nodes[i];
            if (ggml_is_empty(dst)) {
                continue;
            }

            if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW) {
                // TODO: remove GGML_OP_VIEW after view op is supported
                continue;
            }

            QNN_LOG_DEBUG("[%s]create op: %s", get_backend_name(_device), get_qnn_op_name(dst->op));
            auto operation = create_operation_from_op_tensor(dst, dst->name, rank, _device, _graph_handle,
                                                             _qnn_instance, true, tensor_cache); // TODO: fix op name
            operations.push_back(operation);
        }

        _tensor_inputs = std::move(input_tensors);
        _tensor_outputs = std::move(output_tensors);
        _operations = std::move(operations);
        if (!finalize()) {
            return false;
        }
    }

    QNN_LOG_DEBUG("[%s][%s]build succeed", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::execute(ggml_tensor *op) {
    if (!bind_src_tensors(op, _tensor_inputs, _qnn_tensor_inputs)) {
        QNN_LOG_ERROR("[%s][%s]bind input tensors failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    if (!qnn::bind_tensors({op}, _tensor_outputs, _qnn_tensor_outputs)) {
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

bool qnn_graph::execute(const ggml_cgraph *cgraph) {
    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
#ifdef NDEBUG
    get_io_tensors_from_graph(cgraph, inputs, outputs);
#else
    int rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s]rank: %d, input_set: %d, output_set: %d", get_backend_name(_device), rank, int(inputs.size()),
                  int(outputs.size()));
#endif

    {
        if (!qnn::bind_tensors(inputs, _tensor_inputs, _qnn_tensor_inputs)) {
            QNN_LOG_ERROR("[%s][%s]bind input tensors failed", get_backend_name(_device), _graph_name.c_str());
            return false;
        }

        if (!qnn::bind_tensors(outputs, _tensor_outputs, _qnn_tensor_outputs)) {
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
}

bool qnn_graph::finalize() {
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

    QNN_LOG_DEBUG("[%s][%s]finalize succeed", get_backend_name(_device), _graph_name.c_str());
    return true;
}

} // namespace qnn
