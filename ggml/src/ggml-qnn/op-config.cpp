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

struct tensor_common_params {
    const char *name_prefix;
    int tensor_rank;
    QNNBackend device;
    Qnn_GraphHandle_t graph_handle;
    std::shared_ptr<qnn::qnn_instance> qnn_instance;
};

void create_tensors_from_ggml_tensor(const tensor_common_params &params, bool is_input,
                                     const qnn::ggml_tensor_array_t &ggml_tensors,
                                     qnn::ggml_qnn_tensor_array_t &tensor_wrappers,
                                     std::vector<Qnn_Tensor_t> &qnn_tensors) {
    using namespace qnn;

    tensor_wrappers.resize(ggml_tensors.size());
    qnn_tensors.resize(ggml_tensors.size());
    char buffer[GGML_MAX_NAME] = {};
    auto tensor_type = is_input ? ggml_qnn_tensor::INPUT : ggml_qnn_tensor::OUTPUT;
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        snprintf(buffer, GGML_MAX_NAME, "%s%d", params.name_prefix, (int)i);
        auto *ggml_tensor = ggml_tensors[i];
        tensor_wrappers[i] = std::make_shared<ggml_qnn_tensor>(tensor_type, std::string(buffer), ggml_tensor->ne,
                                                               ggml_tensor->type, params.tensor_rank, params.device,
                                                               params.graph_handle, params.qnn_instance);
    }
}

bool bind_tensors(const qnn::ggml_tensor_array_t &ggml_tensors, qnn::ggml_qnn_tensor_array_t &tensor_wrappers,
                  std::vector<Qnn_Tensor_t> &qnn_tensors) {
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        auto *ggml_tensor = ggml_tensors[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor)) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }

        qnn_tensors[i] = tensor_wrappers[i]->get_qnn_tensor();
    }

    return true;
}

class ggml_qnn_connectable_op_config : public qnn::ggml_qnn_op_config_base {
public:
    explicit ggml_qnn_connectable_op_config(const std::string &name, const std::string &package_name,
                                            const std::string &op_type) :
        ggml_qnn_op_config_base(name, package_name, op_type) {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                        std::shared_ptr<qnn::qnn_instance> qnn_instance, const qnn::ggml_tensor_array_t &tensor_inputs,
                        const qnn::ggml_tensor_array_t &tensor_outputs) override {
        GGML_UNUSED(device);
        GGML_UNUSED(graph_handle);
        GGML_UNUSED(qnn_instance);
        GGML_UNUSED(tensor_inputs);
        GGML_UNUSED(tensor_outputs);
        return true;
    }

    void set_input_tensors(qnn::ggml_qnn_tensor_array_t &tensor_inputs) { _tensor_inputs = tensor_inputs; }
    void set_output_tensors(qnn::ggml_qnn_tensor_array_t &tensor_outputs) { _tensor_outputs = tensor_outputs; }

    qnn::ggml_qnn_tensor_array_t &get_input_tensors() { return _tensor_inputs; }
    qnn::ggml_qnn_tensor_array_t &get_output_tensors() { return _tensor_outputs; }

private:
    DISABLE_COPY(ggml_qnn_connectable_op_config);
    DISABLE_MOVE(ggml_qnn_connectable_op_config);
};

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

bool ggml_qnn_op_config_base::bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) {
    GGML_ASSERT(tensor_inputs.size() == _tensor_inputs.size());
    return bind_tensors(tensor_inputs, _tensor_inputs, _qnn_tensor_inputs);
}

bool ggml_qnn_op_config_base::bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) {
    GGML_ASSERT(tensor_outputs.size() == _tensor_outputs.size());
    return bind_tensors(tensor_outputs, _tensor_outputs, _qnn_tensor_outputs);
}

void ggml_qnn_op_config_base::unbind_input_tensors() {
    for (auto &tensor : _tensor_inputs) {
        tensor->unbind();
    }
}

void ggml_qnn_op_config_base::unbind_output_tensors() {
    for (auto &tensor : _tensor_outputs) {
        tensor->unbind();
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
    tensor_common_params params = { "src", tensor_rank, device, graph_handle, qnn_instance };
    create_tensors_from_ggml_tensor(params, true, tensor_inputs, _tensor_inputs, _qnn_tensor_inputs);
    params.name_prefix = "dst";
    create_tensors_from_ggml_tensor(params, false, tensor_outputs, _tensor_outputs, _qnn_tensor_outputs);
    return true;
}

bool ggml_qnn_matmul_op_config::create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               std::shared_ptr<qnn_instance> qnn_instance,
                                               const ggml_tensor_array_t &tensor_inputs,
                                               const ggml_tensor_array_t &tensor_outputs) {
    const auto tensor_rank = get_rank(tensor_inputs, tensor_outputs);
    tensor_common_params params = { "src", tensor_rank, device, graph_handle, qnn_instance };
    create_tensors_from_ggml_tensor(params, true, tensor_inputs, _tensor_inputs, _qnn_tensor_inputs);

    // create intermediate tensor
    auto *first_ggml_tensor = tensor_inputs.front();
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS does not match the expected value");
    ggml_qnn_dimension_array_t dimensions = {
        first_ggml_tensor->ne[1],
        first_ggml_tensor->ne[0],
        first_ggml_tensor->ne[2],
        first_ggml_tensor->ne[3],
    };
    auto intermediate_tensor =
        std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, "intermediate", dimensions,
                                          first_ggml_tensor->type, tensor_rank, device, graph_handle, qnn_instance);

    // create mat_mul
    auto mat_mul =
        std::make_shared<ggml_qnn_connectable_op_config>(_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL);
    params.name_prefix = "dst";
    create_tensors_from_ggml_tensor(params, false, tensor_outputs, mat_mul->get_output_tensors(),
                                    mat_mul->get_qnn_output_tensors());

    // create transpose
    auto transpose = std::make_shared<ggml_qnn_connectable_op_config>(_name + "_trans", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                      QNN_OP_TRANSPOSE);

    // set transpose parameters
    transpose->add_scalar_param("perm", QNN_SCALAR_INIT);

    // set tensor to transpose and mat_mul
    // the graph here will look like:
    // src0 -> | transpose | -> intermediate -> | mat_mul | -> dst0
    //                                  src1 -> | mat_mul |
    ggml_qnn_tensor_array_t tensors = { _tensor_inputs.front() };
    transpose->set_input_tensors(tensors);
    tensors = { intermediate_tensor };
    transpose->set_output_tensors(tensors);
    tensors = { intermediate_tensor, _tensor_inputs.back() };
    mat_mul->set_input_tensors(tensors);

    _mat_mul = mat_mul;
    _transpose = transpose;
    return true;
}

bool ggml_qnn_matmul_op_config::add_op_to_graph(Qnn_GraphHandle_t graph_handle,
                                                std::shared_ptr<qnn_instance> qnn_instance) {
    return _transpose->add_op_to_graph(graph_handle, qnn_instance) &&
           _mat_mul->add_op_to_graph(graph_handle, qnn_instance);
}

bool ggml_qnn_matmul_op_config::bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) {
    return bind_tensors(tensor_inputs, _tensor_inputs, _qnn_tensor_inputs);
}

bool ggml_qnn_matmul_op_config::bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) {
    return _mat_mul->bind_output_tensors(tensor_outputs);
}

void ggml_qnn_matmul_op_config::unbind_input_tensors() {
    _transpose->unbind_input_tensors();
    _mat_mul->unbind_input_tensors();
}

void ggml_qnn_matmul_op_config::unbind_output_tensors() {
    _transpose->unbind_output_tensors();
    _mat_mul->unbind_output_tensors();
}

} // namespace qnn
