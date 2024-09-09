#pragma once

#include <string>
#include <vector>

#include "ggml-qnn.h"

#include "logger.hpp"
#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {
class ggml_qnn_op_config {
public:
    explicit ggml_qnn_op_config(const std::string &name, const std::string &package_name, const std::string &op_type) :
        _name(name), _package_name(package_name), _op_type(op_type) {}

    void add_scalar_param(const std::string &name, const Qnn_Scalar_t scalar) {
        _param_names.push_back(name);
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.paramType = QNN_PARAMTYPE_SCALAR;
        param.name = _param_names.back().c_str();
        param.scalarParam = scalar;
        _parameters.push_back(param);
    }

    void create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance,
                        const size_t input_count, const size_t output_count) {
        _tensor_inputs.resize(input_count);
        _qnn_tensor_inputs.resize(input_count);
        char buffer[GGML_MAX_NAME] = {};
        for (size_t i = 0; i < input_count; i++) {
            snprintf(buffer, GGML_MAX_NAME, "src%d", (int)i);
            _tensor_inputs[i] =
                std::make_shared<ggml_qnn_tensor>(std::string(buffer), device, graph_handle, qnn_instance);
        }

        _tensor_outputs.resize(output_count);
        _qnn_tensor_outputs.resize(output_count);
        for (size_t i = 0; i < output_count; i++) {
            snprintf(buffer, GGML_MAX_NAME, "dst%d", (int)i);
            _tensor_outputs[i] =
                std::make_shared<ggml_qnn_tensor>(std::string(buffer), device, graph_handle, qnn_instance);
        }
    }

    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() { return _qnn_tensor_outputs; }

    Qnn_OpConfig_t get_op_config() {
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

    bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) {
        GGML_ASSERT(tensor_inputs.size() == _tensor_inputs.size());
        GGML_ASSERT(tensor_outputs.size() == _tensor_outputs.size());

        int tensor_rank = 0;

        // get the max tensor rank
        for (auto tensor : tensor_inputs) {
            tensor_rank = std::max(tensor_rank, ggml_n_dims(tensor));
        }
        for (auto tensor : tensor_outputs) {
            tensor_rank = std::max(tensor_rank, ggml_n_dims(tensor));
        }

        for (size_t i = 0; i < tensor_inputs.size(); i++) {
            auto *ggml_tensor = tensor_inputs[i];
            if (!_tensor_inputs[i]->bind_ggml_tensor(ggml_tensor, true, tensor_rank)) {
                QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
                return false;
            }

            _qnn_tensor_inputs[i] = _tensor_inputs[i]->get_qnn_tensor();
        }

        for (size_t i = 0; i < tensor_outputs.size(); i++) {
            auto *ggml_tensor = tensor_outputs[i];
            if (!_tensor_outputs[i]->bind_ggml_tensor(ggml_tensor, false, tensor_rank)) {
                QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
                return false;
            }

            _qnn_tensor_outputs[i] = _tensor_outputs[i]->get_qnn_tensor();
        }

        return true;
    }

    void unbind_tensors() {
        for (auto tensor : _tensor_inputs) {
            tensor->unbind_ggml_tensor();
        }

        for (auto tensor : _tensor_outputs) {
            tensor->unbind_ggml_tensor();
        }
    }

private:
    std::string _name;
    std::string _package_name;
    std::string _op_type;
    std::vector<std::shared_ptr<ggml_qnn_tensor>> _tensor_inputs;
    std::vector<std::shared_ptr<ggml_qnn_tensor>> _tensor_outputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;
    std::vector<Qnn_Param_t> _parameters;
    std::vector<std::string> _param_names;

    DISABLE_COPY(ggml_qnn_op_config);
    DISABLE_MOVE(ggml_qnn_op_config);
};
} // namespace qnn
