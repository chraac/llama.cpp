#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "op-config-base.hpp"
#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {

using ggml_op_constructor_t =
    std::function<std::unique_ptr<ggml_qnn_op_config>(const std::string &, std::shared_ptr<qnn_instance>)>;

ggml_op_constructor_t create_op_constructor(const std::string &op_name);

class ggml_qnn_op_config_base : public ggml_qnn_op_config {
public:
    explicit ggml_qnn_op_config_base(const std::string &name, const std::string &package_name,
                                     const std::string &op_type, std::shared_ptr<qnn_instance> qnn_instance)
        : _name(name), _package_name(package_name), _op_type(op_type), _qnn_instance(qnn_instance) {}

    void add_scalar_param(const std::string &name, const Qnn_Scalar_t scalar);
    bool add_tensor_param(const std::string &name, const qnn_dimension_array_t &dimensions, int rank,
                          const uint8_t *data, const Qnn_DataType_t data_type, QNNBackend device,
                          Qnn_GraphHandle_t graph_handle);
    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) override;
    bool bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) override;
    bool bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) override;
    void unbind_input_tensors() override;
    void unbind_output_tensors() override;
    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override { return _qnn_tensor_outputs; }

protected:
    Qnn_OpConfig_t get_op_config();

    std::string _name;
    std::string _package_name;
    std::string _op_type;
    std::shared_ptr<qnn_instance> _qnn_instance;
    qnn_tensor_array_t _tensor_inputs;
    qnn_tensor_array_t _tensor_outputs;
    qnn_tensor_array_t _tensor_parameters;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;
    std::vector<Qnn_Param_t> _qnn_parameters;
    std::vector<std::string> _param_names;

    DISABLE_COPY(ggml_qnn_op_config_base);
    DISABLE_MOVE(ggml_qnn_op_config_base);
};

class ggml_qnn_single_op_config : public ggml_qnn_op_config_base {
public:
    explicit ggml_qnn_single_op_config(const std::string &name, const std::string &package_name,
                                       const std::string &op_type, std::shared_ptr<qnn_instance> qnn_instance)
        : ggml_qnn_op_config_base(name, package_name, op_type, qnn_instance) {}

    explicit ggml_qnn_single_op_config(const std::string &name, const std::string &package_name,
                                       const std::string &op_type, const std::string &param_name,
                                       const Qnn_DataType_t param_type, const size_t param_size,
                                       std::shared_ptr<qnn_instance> qnn_instance)
        : ggml_qnn_op_config_base(name, package_name, op_type, qnn_instance),
          _param_name(param_name),
          _param_type(param_type),
          _param_buffer(param_size) {}

    bool initialize_op_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                             const ggml_tensor_array_t &tensor_inputs,
                             const ggml_tensor_array_t &tensor_outputs) override;

private:
    const std::string _param_name;
    const Qnn_DataType_t _param_type = QNN_DATATYPE_UINT_32;
    std::vector<uint8_t> _param_buffer;

    DISABLE_COPY(ggml_qnn_single_op_config);
    DISABLE_MOVE(ggml_qnn_single_op_config);
};

class ggml_qnn_aggregate_op_config : public ggml_qnn_op_config {
public:
    explicit ggml_qnn_aggregate_op_config(const std::string &name, std::shared_ptr<qnn_instance> qnn_instance)
        : _name(name), _qnn_instance(qnn_instance) {}

    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) override {
        for (auto &op : _operations) {
            if (!op->add_op_to_graph(graph_handle)) {
                return false;
            }
        }
        return true;
    }

    bool bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) override;

    bool bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) override;

    void unbind_input_tensors() override {
        for (auto &tensor : _tensor_inputs) {
            tensor->unbind();
        }
    }

    void unbind_output_tensors() override {
        for (auto &tensor : _tensor_outputs) {
            tensor->unbind();
        }
    }

    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override { return _qnn_tensor_outputs; }

protected:
    std::string _name;
    std::shared_ptr<qnn_instance> _qnn_instance;

    std::vector<qnn_op_config_ptr_t> _operations;
    qnn_tensor_array_t _tensor_inputs;
    qnn_tensor_array_t _tensor_outputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;

private:
    DISABLE_COPY(ggml_qnn_aggregate_op_config);
    DISABLE_MOVE(ggml_qnn_aggregate_op_config);
};

class ggml_qnn_matmul_op_config : public ggml_qnn_aggregate_op_config {
public:
    ggml_qnn_matmul_op_config(const std::string &name, std::shared_ptr<qnn_instance> qnn_instance)
        : ggml_qnn_aggregate_op_config(name, qnn_instance) {}

    bool initialize_op_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                             const ggml_tensor_array_t &tensor_inputs,
                             const ggml_tensor_array_t &tensor_outputs) override;

private:
    qnn_tensor_ptr_t create_gather_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                                         qnn_tensor_ptr_t tensor_input, qnn_dimension_array_t output_dimensions);
    bool create_convert_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                              qnn_tensor_array_t &tensor_inputs, qnn_tensor_array_t &tensor_outputs);
    bool create_mat_mul_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                              qnn_tensor_array_t &tensor_inputs, qnn_tensor_array_t &tensor_outputs);

    DISABLE_COPY(ggml_qnn_matmul_op_config);
    DISABLE_MOVE(ggml_qnn_matmul_op_config);
};

} // namespace qnn
