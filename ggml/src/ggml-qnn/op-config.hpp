#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "ggml-qnn.h"

#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {

using ggml_tensor_array_t = std::vector<ggml_tensor *>;

class ggml_qnn_op_config {
public:
    virtual ~ggml_qnn_op_config() {}
    virtual bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                const ggml_tensor_array_t &tensor_inputs,
                                const ggml_tensor_array_t &tensor_outputs) = 0;
    virtual std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() = 0;
    virtual std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() = 0;
    virtual bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) = 0;
    virtual bool bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) = 0;
    virtual bool bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) = 0;
    virtual void unbind_input_tensors() = 0;
    virtual void unbind_output_tensors() = 0;
};

class ggml_qnn_op_config_base : public ggml_qnn_op_config {
public:
    explicit ggml_qnn_op_config_base(const std::string &name, const std::string &package_name,
                                     const std::string &op_type, std::shared_ptr<qnn_instance> qnn_instance) :
        _name(name), _package_name(package_name), _op_type(op_type), _qnn_instance(qnn_instance) {}

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
    ggml_qnn_tensor_array_t _tensor_inputs;
    ggml_qnn_tensor_array_t _tensor_outputs;
    ggml_qnn_tensor_array_t _tensor_parameters;
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
                                       const std::string &op_type, std::shared_ptr<qnn_instance> qnn_instance) :
        ggml_qnn_op_config_base(name, package_name, op_type, qnn_instance) {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, const ggml_tensor_array_t &tensor_inputs,
                        const ggml_tensor_array_t &tensor_outputs) override;

private:
    DISABLE_COPY(ggml_qnn_single_op_config);
    DISABLE_MOVE(ggml_qnn_single_op_config);
};

class ggml_qnn_matmul_op_config : public ggml_qnn_op_config {
public:
    ggml_qnn_matmul_op_config(const std::string &name, std::shared_ptr<qnn_instance> qnn_instance) :
        _name(name), _qnn_instance(qnn_instance) {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, const ggml_tensor_array_t &tensor_inputs,
                        const ggml_tensor_array_t &tensor_outputs) override;
    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) override;
    bool bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) override;
    bool bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) override;
    void unbind_input_tensors() override;
    void unbind_output_tensors() override;
    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override;

private:
    bool create_mat_mul_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                              ggml_qnn_tensor_array_t &tensor_inputs, ggml_qnn_tensor_array_t &tensor_outputs);

    std::string _name;
    std::shared_ptr<qnn_instance> _qnn_instance;
    std::shared_ptr<ggml_qnn_op_config> _transpose0;
    std::shared_ptr<ggml_qnn_op_config> _transpose1;
    std::shared_ptr<ggml_qnn_op_config> _mat_mul;
    std::vector<std::shared_ptr<ggml_qnn_op_config>> _input_converts;
    std::shared_ptr<ggml_qnn_op_config> _output_convert;
    ggml_qnn_tensor_array_t _tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;

    DISABLE_COPY(ggml_qnn_matmul_op_config);
    DISABLE_MOVE(ggml_qnn_matmul_op_config);
};

using ggml_op_constructor_t =
    std::function<std::unique_ptr<ggml_qnn_op_config>(const std::string &, std::shared_ptr<qnn_instance>)>;

ggml_op_constructor_t create_op_constructor(const std::string &op_name);

} // namespace qnn
