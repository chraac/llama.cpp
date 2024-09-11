#pragma once

#include <string>
#include <vector>

#include "ggml-qnn.h"

#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {

class ggml_qnn_op_config {
public:
    virtual ~ggml_qnn_op_config() {}
    virtual bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                std::shared_ptr<qnn_instance> qnn_instance, const ggml_tensor_array_t &tensor_inputs,
                                const ggml_tensor_array_t &tensor_outputs) = 0;
    virtual std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() = 0;
    virtual std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() = 0;
    virtual bool add_nodes(Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance) = 0;
    virtual bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) = 0;
    virtual void unbind_tensors() = 0;
};

class ggml_qnn_single_op_config : public ggml_qnn_op_config {
public:
    explicit ggml_qnn_single_op_config(const std::string &name, const std::string &package_name,
                                       const std::string &op_type) :
        _name(name), _package_name(package_name), _op_type(op_type) {}

    void add_scalar_param(const std::string &name, const Qnn_Scalar_t scalar);
    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance,
                        const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override;
    bool add_nodes(Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance) override;
    bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override;
    void unbind_tensors() override;
    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override { return _qnn_tensor_outputs; }

private:
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

    DISABLE_COPY(ggml_qnn_single_op_config);
    DISABLE_MOVE(ggml_qnn_single_op_config);
};

class ggml_qnn_matmul_op_config : public ggml_qnn_op_config {
public:
    ggml_qnn_matmul_op_config() :
        _transpose("matmul_trans", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE),
        _mat_mul("matmul", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL) {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance,
                        const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override {
        return true;
    }

    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() override { return _transpose.get_qnn_input_tensors(); }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() override { return _mat_mul.get_qnn_output_tensors(); }

    bool add_nodes(Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance) override { return true; }

    bool bind_tensors(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) override {
        return true;
    }

    void unbind_tensors() override {
        _transpose.unbind_tensors();
        _mat_mul.unbind_tensors();
    }

private:
    ggml_qnn_single_op_config _transpose;
    ggml_qnn_single_op_config _mat_mul;

    DISABLE_COPY(ggml_qnn_matmul_op_config);
    DISABLE_MOVE(ggml_qnn_matmul_op_config);
};

} // namespace qnn
