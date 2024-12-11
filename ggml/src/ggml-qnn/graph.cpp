
#include "graph.hpp"

#include "logger.hpp"

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
    _op_config = op_constructor(_graph_name, _qnn_instance);
    if (!_op_config->initialize_op_nodes(_device, _graph_handle, tensor_inputs, tensor_outputs)) {
        QNN_LOG_ERROR("[%s][%s]initialize_op_nodes failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    if (!_op_config->add_op_to_graph(_graph_handle)) {
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
    if (!_op_config->bind_input_tensors(tensor_inputs)) {
        QNN_LOG_ERROR("[%s][%s]bind input tensors failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    if (!_op_config->bind_output_tensors(tensor_outputs)) {
        QNN_LOG_ERROR("[%s][%s]bind output tensors failed", get_backend_name(_device), _graph_name.c_str());
        return false;
    }

    auto &qnn_tensor_inputs = _op_config->get_qnn_input_tensors();
    auto &qnn_tensor_outputs = _op_config->get_qnn_output_tensors();

    auto error =
        _qnn_interface->qnn_graph_execute(_graph_handle, qnn_tensor_inputs.data(), qnn_tensor_inputs.size(),
                                          qnn_tensor_outputs.data(), qnn_tensor_outputs.size(), nullptr, nullptr);
    _op_config->unbind_input_tensors();
    _op_config->unbind_output_tensors();

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

} // namespace qnn
