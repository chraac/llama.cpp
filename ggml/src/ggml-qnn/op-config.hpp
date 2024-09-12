#pragma once

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
                                std::shared_ptr<qnn_instance> qnn_instance, const ggml_tensor_array_t &tensor_inputs,
                                const ggml_tensor_array_t &tensor_outputs) = 0;
    virtual std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() = 0;
    virtual std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() = 0;
    virtual bool add_op_to_graph(Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance) = 0;
    virtual bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) = 0;
    virtual void unbind_tensors() = 0;
};

class ggml_qnn_op_config_base : public ggml_qnn_op_config {
public:
    explicit ggml_qnn_op_config_base(const std::string &name, const std::string &package_name,
                                     const std::string &op_type) :
        _name(name), _package_name(package_name), _op_type(op_type) {}

    void add_scalar_param(const std::string &name, const Qnn_Scalar_t scalar);
    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance) override;
    bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override;
    void unbind_tensors() override;
    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override { return _qnn_tensor_outputs; }

protected:
    Qnn_OpConfig_t get_op_config();

    std::string _name;
    std::string _package_name;
    std::string _op_type;
    std::vector<std::shared_ptr<ggml_qnn_tensor>> _tensor_inputs;
    std::vector<std::shared_ptr<ggml_qnn_tensor>> _tensor_outputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;
    std::vector<Qnn_Param_t> _parameters;
    std::vector<std::string> _param_names;

    DISABLE_COPY(ggml_qnn_op_config_base);
    DISABLE_MOVE(ggml_qnn_op_config_base);
};

class ggml_qnn_single_op_config : public ggml_qnn_op_config_base {
public:
    explicit ggml_qnn_single_op_config(const std::string &name, const std::string &package_name,
                                       const std::string &op_type) :
        ggml_qnn_op_config_base(name, package_name, op_type) {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance,
                        const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override;

private:
    DISABLE_COPY(ggml_qnn_single_op_config);
    DISABLE_MOVE(ggml_qnn_single_op_config);
};

class ggml_qnn_matmul_op_config : public ggml_qnn_op_config {
public:
    ggml_qnn_matmul_op_config() {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance,
                        const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override;
    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _transpose->get_qnn_input_tensors(); }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override { return _mat_mul->get_qnn_output_tensors(); }

    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance) override {
        return _transpose->add_op_to_graph(graph_handle, qnn_instance) &&
               _mat_mul->add_op_to_graph(graph_handle, qnn_instance);
    }

    bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override {
        return _transpose->bind_tensors(tensor_inputs, tensor_outputs) &&
               _mat_mul->bind_tensors(tensor_inputs, tensor_outputs);
    }

    void unbind_tensors() override {
        _transpose->unbind_tensors();
        _mat_mul->unbind_tensors();
    }

private:
    std::unique_ptr<ggml_qnn_op_config> _transpose;
    std::unique_ptr<ggml_qnn_op_config> _mat_mul;

    DISABLE_COPY(ggml_qnn_matmul_op_config);
    DISABLE_MOVE(ggml_qnn_matmul_op_config);
};

} // namespace qnn
