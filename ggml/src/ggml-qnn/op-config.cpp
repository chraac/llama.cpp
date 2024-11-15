#include "op-config.hpp"

#include <cstdint>

#include "logger.hpp"

namespace {

constexpr const qnn::qnn_dimension_array_t kTransposeParamData[GGML_MAX_DIMS] = {
    {0},
    {1, 0},
    {0, 2, 1},
    {0, 1, 3, 2},
};

qnn::qnn_dimension_array_t get_transposed_dimensions(const qnn::qnn_dimension_array_t &dimensions, int rank) {
    qnn::qnn_dimension_array_t transposed_dims = dimensions;
    if (rank >= 2) {
        transposed_dims[rank - 1] = dimensions[rank - 2];
        transposed_dims[rank - 2] = dimensions[rank - 1];
    }

    return transposed_dims;
}

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

Qnn_DataType_t get_tensor_type(const qnn::ggml_qnn_tensor_array_t &tensors) {
    Qnn_DataType_t type = QNN_DATATYPE_UNDEFINED;
    for (auto tensor : tensors) {
        auto tensor_type_size = qnn::qnn_datatype_size(tensor->get_data_type());
        GGML_ASSERT(tensor_type_size > 0);
        if (tensor_type_size > qnn::qnn_datatype_size(type)) {
            type = tensor->get_data_type();
        }
    }

    return type;
}

struct tensor_common_params {
    const char *name_prefix;
    int tensor_rank;
    bool is_input;
    QNNBackend device;
    Qnn_GraphHandle_t graph_handle;
    std::shared_ptr<qnn::qnn_instance> qnn_instance;
};

void create_tensors_from_ggml_tensor(const tensor_common_params &params, const qnn::ggml_tensor_array_t &ggml_tensors,
                                     qnn::ggml_qnn_tensor_array_t *tensor_wrappers,
                                     std::vector<Qnn_Tensor_t> *qnn_tensors) {
    using namespace qnn;

    tensor_wrappers->resize(ggml_tensors.size());
    if (qnn_tensors) {
        qnn_tensors->resize(ggml_tensors.size());
    }
    char buffer[GGML_MAX_NAME] = {};
    auto tensor_type = params.is_input ? ggml_qnn_tensor::INPUT : ggml_qnn_tensor::OUTPUT;
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        snprintf(buffer, GGML_MAX_NAME, "%s%d", params.name_prefix, (int)i);
        auto *ggml_tensor = ggml_tensors[i];
        (*tensor_wrappers)[i] = std::make_shared<ggml_qnn_tensor>(tensor_type, std::string(buffer), ggml_tensor->ne,
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
                                            const std::string &op_type, std::shared_ptr<qnn::qnn_instance> qnn_instance)
        : ggml_qnn_op_config_base(name, package_name, op_type, qnn_instance) {}

    bool create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                        const qnn::ggml_tensor_array_t &tensor_inputs,
                        const qnn::ggml_tensor_array_t &tensor_outputs) override {
        GGML_UNUSED(device);
        GGML_UNUSED(graph_handle);
        GGML_UNUSED(tensor_inputs);
        GGML_UNUSED(tensor_outputs);
        return true;
    }

    void set_input_tensors(qnn::ggml_qnn_tensor_array_t &tensor_inputs) {
        _tensor_inputs = tensor_inputs;
        _qnn_tensor_inputs.resize(_tensor_inputs.size());
    }

    void set_input_tensors(qnn::ggml_qnn_tensor_array_t &&tensor_inputs) {
        _tensor_inputs = std::move(tensor_inputs);
        _qnn_tensor_inputs.resize(_tensor_inputs.size());
    }

    void set_output_tensors(qnn::ggml_qnn_tensor_array_t &tensor_outputs) {
        _tensor_outputs = tensor_outputs;
        _qnn_tensor_outputs.resize(_tensor_outputs.size());
    }

    void set_output_tensors(qnn::ggml_qnn_tensor_array_t &&tensor_outputs) {
        _tensor_outputs = std::move(tensor_outputs);
        _qnn_tensor_outputs.resize(_tensor_outputs.size());
    }

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
    _qnn_parameters.push_back(param);
}

bool ggml_qnn_op_config_base::add_tensor_param(const std::string &name, const qnn_dimension_array_t &dimensions,
                                               int rank, const uint8_t *data, const Qnn_DataType_t data_type,
                                               QNNBackend device, Qnn_GraphHandle_t graph_handle) {
    std::string tensor_name = _name + name + std::to_string(_tensor_parameters.size());
    auto param_tensor = std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::PARAMETER, tensor_name, dimensions,
                                                          data_type, rank, device, graph_handle, _qnn_instance);
    size_t data_size = ggml_type_size(ggml_datatype_from_qnn_datatype(data_type));
    for (int i = 0; i < rank; i++) {
        data_size *= dimensions[i];
    }

    GGML_ASSERT(data_size > 0);
    if (!param_tensor->bind_buffer(const_cast<uint8_t *>(data), data_size)) {
        QNN_LOG_ERROR("parameter tensor bind_buffer failed\n");
        return false;
    }

    if (!param_tensor->alloc_qnn_tensor_id()) {
        QNN_LOG_ERROR("parameter tensor alloc_qnn_tensor_id failed\n");
        return false;
    }

    _tensor_parameters.push_back(param_tensor);
    _param_names.push_back(name);
    Qnn_Param_t param = QNN_PARAM_INIT;
    param.paramType = QNN_PARAMTYPE_TENSOR;
    param.name = _param_names.back().c_str();
    param.tensorParam = param_tensor->get_qnn_tensor();
    _qnn_parameters.push_back(param);
    return true;
}

bool ggml_qnn_op_config_base::add_op_to_graph(Qnn_GraphHandle_t graph_handle) {
    GGML_ASSERT(_qnn_tensor_inputs.size() == _tensor_inputs.size());
    GGML_ASSERT(_qnn_tensor_outputs.size() == _tensor_outputs.size());

    auto qnn_interface = _qnn_instance->get_qnn_interface();
    for (size_t i = 0; i < _tensor_inputs.size(); i++) {
        auto tensor = _tensor_inputs[i];
        if (!tensor->alloc_qnn_tensor_id()) {
            QNN_LOG_ERROR("[%s]input tensor alloc_qnn_tensor_id failed\n", _name.c_str());
            return false;
        }

        _qnn_tensor_inputs[i] = tensor->get_qnn_tensor();
    }

    for (size_t i = 0; i < _tensor_outputs.size(); i++) {
        auto tensor = _tensor_outputs[i];
        if (!tensor->alloc_qnn_tensor_id()) {
            QNN_LOG_ERROR("[%s]output tensor alloc_qnn_tensor_id failed\n", _name.c_str());
            return false;
        }
        _qnn_tensor_outputs[i] = _tensor_outputs[i]->get_qnn_tensor();
    }

    auto error = qnn_interface->qnn_graph_add_node(graph_handle, get_op_config());
    if (error != QNN_SUCCESS) {
        auto *error_str = get_qnn_error_string(error);
        if (error_str) {
            QNN_LOG_ERROR("[%s]qnn_graph_add_node.error: %s\n", _name.c_str(), error_str);
        } else {
            QNN_LOG_ERROR("[%s]qnn_graph_add_node.error: %d\n", _name.c_str(), error);
        }
        return false;
    }

    QNN_LOG_DEBUG("[%s]added to graph\n", _name.c_str());
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
    op_config.numOfParams = (uint32_t)_qnn_parameters.size();
    op_config.params = _qnn_parameters.data();
    op_config.numOfInputs = (uint32_t)_qnn_tensor_inputs.size();
    op_config.inputTensors = _qnn_tensor_inputs.data();
    op_config.numOfOutputs = (uint32_t)_qnn_tensor_outputs.size();
    op_config.outputTensors = _qnn_tensor_outputs.data();
    return config;
}

bool ggml_qnn_single_op_config::create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               const ggml_tensor_array_t &tensor_inputs,
                                               const ggml_tensor_array_t &tensor_outputs) {
    const auto tensor_rank = get_rank(tensor_inputs, tensor_outputs);
    tensor_common_params params = {"src", tensor_rank, true, device, graph_handle, _qnn_instance};
    create_tensors_from_ggml_tensor(params, tensor_inputs, &_tensor_inputs, &_qnn_tensor_inputs);
    params.name_prefix = "dst";
    params.is_input = false;
    create_tensors_from_ggml_tensor(params, tensor_outputs, &_tensor_outputs, &_qnn_tensor_outputs);

    if (_param_buffer.size() > 0) {
        // handle parameters in output tensor
        auto *params = tensor_outputs.front()->op_params;
        memcpy(_param_buffer.data(), params, _param_buffer.size());

        const uint32_t count = uint32_t(_param_buffer.size() / qnn_datatype_size(_param_type));
        const qnn_dimension_array_t param_dims = {count, 1, 1, 1};
        add_tensor_param(_param_name, param_dims, 1, _param_buffer.data(), _param_type, device, graph_handle);
    }

    return true;
}

bool ggml_qnn_matmul_op_config::create_tensors(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               const ggml_tensor_array_t &tensor_inputs,
                                               const ggml_tensor_array_t &tensor_outputs) {
    GGML_ASSERT(tensor_inputs.size() == 2);
    GGML_ASSERT(tensor_outputs.size() == 1);
    const auto tensor_rank = get_rank(tensor_inputs, tensor_outputs);
    GGML_ASSERT(tensor_rank >= 2);

    // create input tensors
    tensor_common_params params = {"src", tensor_rank, true, device, graph_handle, _qnn_instance};
    create_tensors_from_ggml_tensor(params, tensor_inputs, &_tensor_inputs, &_qnn_tensor_inputs);

    // create output tensor
    ggml_qnn_tensor_array_t mat_mul_tensor_outputs;
    params.name_prefix = "dst";
    params.is_input = false;
    create_tensors_from_ggml_tensor(params, tensor_outputs, &mat_mul_tensor_outputs, nullptr);

    ggml_qnn_tensor_array_t mat_mul_tensor_inputs = _tensor_inputs;
    mat_mul_tensor_inputs.front() = create_resize_node(device, graph_handle, tensor_rank, _tensor_inputs.front(),
                                                       _tensor_inputs.back()->get_dimensions());
    return create_mat_mul_nodes(device, graph_handle, tensor_rank, mat_mul_tensor_inputs, mat_mul_tensor_outputs);
}

bool ggml_qnn_matmul_op_config::create_mat_mul_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                                                     ggml_qnn_tensor_array_t &tensor_inputs,
                                                     ggml_qnn_tensor_array_t &tensor_outputs) {

    /*
     * First, both the ggml and qnn tensor in memory are stored as row-major format. (For more details, please also:
     * https://pytorch.org/blog/tensor-memory-format-matters/#:~:text=Column%20Major%20Order:%20In%20this%20format,%20the%20matrix)
     * But the dimensions of the tensor are stored in different order.
     * For example, a 2x3 matrix:
     *   [
     *     [1, 2, 3],
     *     [4, 5, 6],
     *   ]
     * The ggml tensor will have dimensions [3, 2], while the qnn tensor will have dimensions [2, 3].
     *
     * Second, from the ggml introduction here: https://github.com/huggingface/blog/blob/main/introduction-to-ggml.md
     * Given 2 matrices A and B, the matrix multiplication C = A * B is defined as:
     * ```python
     * import torch
     * # Create two matrices
     * A = torch.tensor([
     *   [2, 8],
     *   [5, 1],
     *   [4, 2],
     *   [8, 6],
     * ])
     * B = torch.tensor([
     *   [10, 5],
     *   [9, 9],
     *   [5, 4],
     * ])
     * # Perform matrix multiplication
     * result = torch.matmul(A, B.T)
     * print(result.T)
     * ```
     * Here, the B.T is the transpose of B.
     *
     * So here we need to create graph like:
     *   ```mermaid
     *   graph TD;
     *        i1>ggml_tensor_in0] --src0--> mat_mul0;
     *        i2>ggml_tensor_in1] --src1--> transpose0;
     *        transpose0 --src0_trans--> mat_mul0;
     *        mat_mul0 --dst_trans--> transpose1;
     *        transpose1 --dst0--> o1>ggml_tensor_out];
     *   ```
     */

    // create src0_trans tensor
    auto src1 = tensor_inputs.back();
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS does not match the expected value");

    qnn_dimension_array_t dimensions = get_transposed_dimensions(src1->get_dimensions(), rank);
    auto src0_trans =
        std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, "src0_trans", dimensions,
                                          src1->get_data_type(), rank, device, graph_handle, _qnn_instance);

    // create dst_trans tensor
    auto dst = tensor_outputs.front();
    dimensions = get_transposed_dimensions(dst->get_dimensions(), rank);
    auto dst_trans = std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, "dst_trans", dimensions,
                                                       dst->get_data_type(), rank, device, graph_handle, _qnn_instance);

    // create transpose0
    auto transpose0 = std::make_shared<ggml_qnn_connectable_op_config>(_name + "_trans0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                       QNN_OP_TRANSPOSE, _qnn_instance);

    // create transpose1
    auto transpose1 = std::make_shared<ggml_qnn_connectable_op_config>(_name + "_trans1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                       QNN_OP_TRANSPOSE, _qnn_instance);

    // create mat_mul
    auto mat_mul = std::make_shared<ggml_qnn_connectable_op_config>(_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                                                    _qnn_instance);

    // set transpose0 parameters
    auto *params_data = reinterpret_cast<const uint8_t *>(kTransposeParamData[rank - 1].data());
    const qnn_dimension_array_t param_dims = {(uint32_t)rank, 1, 1, 1};
    transpose0->add_tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, param_dims, 1, params_data, QNN_DATATYPE_UINT_32, device,
                                 graph_handle);

    // set transpose1 parameters
    transpose1->add_tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, param_dims, 1, params_data, QNN_DATATYPE_UINT_32, device,
                                 graph_handle);

    // set tensor to transpose0
    ggml_qnn_tensor_array_t tensors = {tensor_inputs.back()};
    transpose0->set_input_tensors(tensors);
    tensors = {src0_trans};
    transpose0->set_output_tensors(tensors);

    // set tensor to mat_mul
    tensors = {tensor_inputs.front(), src0_trans};
    mat_mul->set_input_tensors(tensors);
    tensors = {dst_trans};
    mat_mul->set_output_tensors(tensors);

    // set tensor to transpose1
    tensors = {dst_trans};
    transpose1->set_input_tensors(tensors);
    transpose1->set_output_tensors(tensor_outputs);

    _mat_mul = mat_mul;
    _transpose0 = transpose0;
    _transpose1 = transpose1;
    return true;
}

ggml_qnn_tensor_ptr_t ggml_qnn_matmul_op_config::create_resize_node(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                                                    const int rank, ggml_qnn_tensor_ptr_t tensor_input,
                                                                    qnn_dimension_array_t output_dimensions) {
    if (rank <= 2) {
        return tensor_input;
    }

    const auto &input_dimensions = tensor_input->get_dimensions();
    output_dimensions[rank - 1] = input_dimensions[rank - 1];
    output_dimensions[rank - 2] = input_dimensions[rank - 2];
    // create concat nodes, to convert tensor shape from [ne03, ne02, n, k] to [ne03 * x, ne02 * y, n, k]
    const auto y = output_dimensions[rank - 3] / input_dimensions[rank - 3];
    if (y == 1 && rank == 3) {
        return tensor_input;
    }

    std::string name = _name + "_resize";
    auto resize_out =
        std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, name + "_out", output_dimensions,
                                          tensor_input->get_data_type(), rank, device, graph_handle, _qnn_instance);
    auto resize_op = std::make_shared<ggml_qnn_connectable_op_config>(name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESIZE,
                                                                      _qnn_instance);
    resize_op->set_input_tensors({tensor_input});
    resize_op->set_output_tensors({resize_out});

    Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
    scalar.dataType = QNN_DATATYPE_UINT_32;
    scalar.uint32Value = 3; // ASYMMETRIC = 3
    resize_op->add_scalar_param(QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE, scalar);
    _resize = resize_op;
    return resize_out;
}

bool ggml_qnn_matmul_op_config::add_op_to_graph(Qnn_GraphHandle_t graph_handle) {
    for (auto &convert : _input_converts) {
        if (convert && !convert->add_op_to_graph(graph_handle)) {
            return false;
        }
    }

    if (_resize && !_resize->add_op_to_graph(graph_handle)) {
        return false;
    }

    return _transpose0->add_op_to_graph(graph_handle) && _mat_mul->add_op_to_graph(graph_handle) &&
           _transpose1->add_op_to_graph(graph_handle) &&
           (!_output_convert || _output_convert->add_op_to_graph(graph_handle));
}

bool ggml_qnn_matmul_op_config::bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) {
    return bind_tensors(tensor_inputs, _tensor_inputs, _qnn_tensor_inputs);
}

bool ggml_qnn_matmul_op_config::bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) {
    if (_output_convert) {
        return _output_convert->bind_output_tensors(tensor_outputs);
    } else {
        return _transpose1->bind_output_tensors(tensor_outputs);
    }
}

void ggml_qnn_matmul_op_config::unbind_input_tensors() {
    _mat_mul->unbind_input_tensors();
    _transpose0->unbind_input_tensors();
    for (auto &convert : _input_converts) {
        if (convert) {
            convert->unbind_input_tensors();
        }
    }
}

void ggml_qnn_matmul_op_config::unbind_output_tensors() {
    _transpose1->unbind_output_tensors();
    if (_output_convert) {
        _output_convert->unbind_output_tensors();
    }
}

std::vector<Qnn_Tensor_t> &ggml_qnn_matmul_op_config::get_qnn_output_tensors() {
    if (_output_convert) {
        return _output_convert->get_qnn_output_tensors();
    } else {
        return _transpose1->get_qnn_output_tensors();
    }
}

ggml_op_constructor_t create_op_constructor(const std::string &op_name) {
    if (op_name == QNN_OP_MAT_MUL) {
        // For QNN_OP_MAT_MUL, we need to transpose the input tensor
        return [](const std::string &instance_name,
                  std::shared_ptr<qnn::qnn_instance> qnn_instance) -> std::unique_ptr<qnn::ggml_qnn_op_config> {
            QNN_LOG_DEBUG("create QNN_OP_MAT_MUL, name %s\n", instance_name.c_str());
            return std::make_unique<qnn::ggml_qnn_matmul_op_config>(instance_name, qnn_instance);
        };
    } else if (op_name == QNN_OP_TRANSPOSE) {
        return [](const std::string &instance_name,
                  std::shared_ptr<qnn::qnn_instance> qnn_instance) -> std::unique_ptr<qnn::ggml_qnn_op_config> {
            return std::make_unique<qnn::ggml_qnn_single_op_config>(
                instance_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, QNN_OP_TRANSPOSE_PARAM_PERM,
                QNN_DATATYPE_UINT_32, 4 * sizeof(uint32_t), qnn_instance);
        };
    }

    return [op_name](const std::string &instance_name,
                     std::shared_ptr<qnn::qnn_instance> qnn_instance) -> std::unique_ptr<qnn::ggml_qnn_op_config> {
        return std::make_unique<qnn::ggml_qnn_single_op_config>(instance_name, QNN_OP_PACKAGE_NAME_QTI_AISW, op_name,
                                                                qnn_instance);
    };
}

} // namespace qnn
