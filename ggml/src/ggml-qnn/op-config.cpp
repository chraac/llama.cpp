#pragma once

#include "op-config.hpp"

#include "logger.hpp"

namespace {

int get_rank(const qnn::ggml_tensor_array_t &tensor_inputs, const qnn::ggml_tensor_array_t &tensor_outputs) {
    int tensor_rank = 0;
    // get the max tensor rank
    for (auto tensor : tensor_inputs) {
        tensor_rank = std::max(tensor_rank, ggml_n_dims(tensor));
    }
    for (auto tensor : tensor_outputs) {
        tensor_rank = std::max(tensor_rank, ggml_n_dims(tensor));
    }

    return tensor_rank;
}

void create_tensors_from_ggml_tensor(const std::string &prefix, int tensor_rank, bool is_input, QNNBackend device,
                                     Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                     const qnn::ggml_tensor_array_t &ggml_tensors,
                                     std::vector<std::shared_ptr<qnn::ggml_qnn_tensor>> &tensor_wrappers,
                                     std::vector<Qnn_Tensor_t> &qnn_tensors) {
    using namespace qnn;

    tensor_wrappers.resize(ggml_tensors.size());
    qnn_tensors.resize(ggml_tensors.size());
    char buffer[GGML_MAX_NAME] = {};
    auto tensor_type = is_input ? ggml_qnn_tensor::INPUT : ggml_qnn_tensor::OUTPUT;
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        snprintf(buffer, GGML_MAX_NAME, "%s%d", prefix.c_str(), (int)i);
        auto *ggml_tensor = ggml_tensors[i];
        tensor_wrappers[i] =
            std::make_shared<ggml_qnn_tensor>(tensor_type, std::string(buffer), ggml_tensor->ne, ggml_tensor->type,
                                              tensor_rank, device, graph_handle, qnn_instance);
    }
}

} // namespace

namespace qnn {

void ggml_qnn_op_config_base::add_scalar_param(const std::string &name, const Qnn_Scalar_t scalar) {
    _param_names.push_back(name);
    Qnn_Param_t param = QNN_PARAM_INIT;
    param.paramType = QNN_PARAMTYPE_SCALAR;
    param.name = _param_names.back().c_str();
    param.scalarParam = scalar;
    _parameters.push_back(param);
}

bool ggml_qnn_op_config_base::add_op_to_graph(Qnn_GraphHandle_t graph_handle,
                                              std::shared_ptr<qnn_instance> qnn_instance) {
    auto qnn_interface = qnn_instance->get_qnn_interface();

    for (size_t i = 0; i < _tensor_inputs.size(); i++) {
        auto tensor = _tensor_inputs[i];
        if (!tensor->alloc_qnn_tensor_id()) {
            QNN_LOG_ERROR("input tensor alloc_qnn_tensor_id failed\n");
            return false;
        }

        _qnn_tensor_inputs[i] = tensor->get_qnn_tensor();
    }

    for (size_t i = 0; i < _tensor_outputs.size(); i++) {
        auto tensor = _tensor_outputs[i];
        if (!tensor->alloc_qnn_tensor_id()) {
            QNN_LOG_ERROR("output tensor alloc_qnn_tensor_id failed\n");
            return false;
        }
        _qnn_tensor_outputs[i] = _tensor_outputs[i]->get_qnn_tensor();
    }

    auto error = qnn_interface->qnn_graph_add_node(graph_handle, get_op_config());
    if (error != QNN_SUCCESS) {
        auto *error_str = get_qnn_error_string(error);
        if (error_str) {
            QNN_LOG_ERROR("qnn_graph_add_node.error: %s\n", error_str);
        } else {
            QNN_LOG_ERROR("qnn_graph_add_node.error: %d\n", error);
        }
        return false;
    }

    return true;
}

bool ggml_qnn_op_config_base::bind_tensors(const ggml_tensor_array_t &tensor_inputs,
                                           const ggml_tensor_array_t &tensor_outputs) {
    GGML_ASSERT(tensor_inputs.size() == _tensor_inputs.size());
    GGML_ASSERT(tensor_outputs.size() == _tensor_outputs.size());
    for (size_t i = 0; i < tensor_inputs.size(); i++) {
        auto *ggml_tensor = tensor_inputs[i];
        if (!_tensor_inputs[i]->bind_ggml_tensor(ggml_tensor)) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }

        _qnn_tensor_inputs[i] = _tensor_inputs[i]->get_qnn_tensor();
    }

    for (size_t i = 0; i < tensor_outputs.size(); i++) {
        auto *ggml_tensor = tensor_outputs[i];
        if (!_tensor_outputs[i]->bind_ggml_tensor(ggml_tensor)) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }

        _qnn_tensor_outputs[i] = _tensor_outputs[i]->get_qnn_tensor();
    }

    return true;
}

void ggml_qnn_op_config_base::unbind_tensors() {
    for (auto tensor : _tensor_inputs) {
        tensor->unbind_ggml_tensor();
    }

    for (auto tensor : _tensor_outputs) {
        tensor->unbind_ggml_tensor();
    }
}

Qnn_OpConfig_t ggml_qnn_op_config_base::get_op_config() {
    Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;
    config.version = QNN_OPCONFIG_VERSION_1;
    auto &op_config = config.v1;
    op_config.name = _name.c_str();
    op_config.packageName = _package_name.c_str();
    op_config.typeName = _op_type.c_str();
    op_config.numOfParams = (uint32_t)_parameters.size();
    op_config.params = _parameters.data();
    op_config.numOfInputs = (uint32_t)_qnn_tensor_inputs.size();
    op_config.inputTensors = _qnn_tensor_inputs.data();
    op_config.numOfOutputs = (uint32_t)_qnn_tensor_outputs.size();
    op_config.outputTensors = _qnn_tensor_outputs.data();
    return config;
}

bool ggml_qnn_single_op_config::create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               std::shared_ptr<qnn_instance> qnn_instance,
                                               const ggml_tensor_array_t &tensor_inputs,
                                               const ggml_tensor_array_t &tensor_outputs) {
    const auto tensor_rank = get_rank(tensor_inputs, tensor_outputs);
    create_tensors_from_ggml_tensor("src", tensor_rank, true, device, graph_handle, qnn_instance, tensor_inputs,
                                    _tensor_inputs, _qnn_tensor_inputs);
    create_tensors_from_ggml_tensor("dst", tensor_rank, false, device, graph_handle, qnn_instance, tensor_outputs,
                                    _tensor_outputs, _qnn_tensor_outputs);
    return true;
}

bool ggml_qnn_matmul_op_config::create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               std::shared_ptr<qnn_instance> qnn_instance,
                                               const ggml_tensor_array_t &tensor_inputs,
                                               const ggml_tensor_array_t &tensor_outputs) {
    return true;
}

} // namespace qnn
