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

Qnn_DataType_t get_tensor_type(const qnn::qnn_tensor_array_t &tensors) {
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
                                     qnn::qnn_tensor_array_t *tensor_wrappers, std::vector<Qnn_Tensor_t> *qnn_tensors) {
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

bool bind_tensors(const qnn::ggml_tensor_array_t &ggml_tensors, qnn::qnn_tensor_array_t &tensor_wrappers,
                  std::vector<Qnn_Tensor_t> &qnn_tensors) {
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        auto *ggml_tensor = ggml_tensors[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor)) {
            QNN_LOG_ERROR("bind tensor %s failed", ggml_get_name(ggml_tensor));
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

    bool initialize_op_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                             const qnn::ggml_tensor_array_t &tensor_inputs,
                             const qnn::ggml_tensor_array_t &tensor_outputs) override {
        GGML_UNUSED(device);
        GGML_UNUSED(graph_handle);
        GGML_UNUSED(tensor_inputs);
        GGML_UNUSED(tensor_outputs);
        return true;
    }

    void set_input_tensors(qnn::qnn_tensor_array_t &tensor_inputs) {
        _tensor_inputs = tensor_inputs;
        _qnn_tensor_inputs.resize(_tensor_inputs.size());
    }

    void set_input_tensors(qnn::qnn_tensor_array_t &&tensor_inputs) {
        _tensor_inputs = std::move(tensor_inputs);
        _qnn_tensor_inputs.resize(_tensor_inputs.size());
    }

    void set_output_tensors(qnn::qnn_tensor_array_t &tensor_outputs) {
        _tensor_outputs = tensor_outputs;
        _qnn_tensor_outputs.resize(_tensor_outputs.size());
    }

    void set_output_tensors(qnn::qnn_tensor_array_t &&tensor_outputs) {
        _tensor_outputs = std::move(tensor_outputs);
        _qnn_tensor_outputs.resize(_tensor_outputs.size());
    }

    qnn::qnn_tensor_array_t &get_input_tensors() { return _tensor_inputs; }
    qnn::qnn_tensor_array_t &get_output_tensors() { return _tensor_outputs; }

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
        QNN_LOG_ERROR("parameter tensor bind_buffer failed");
        return false;
    }

    if (!param_tensor->alloc_qnn_tensor_id()) {
        QNN_LOG_ERROR("parameter tensor alloc_qnn_tensor_id failed");
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

    QNN_LOG_DEBUG("[%s]add to graph start", _name.c_str());
    for (size_t i = 0; i < _tensor_inputs.size(); i++) {
        auto tensor = _tensor_inputs[i];
        if (!tensor->alloc_qnn_tensor_id()) {
            QNN_LOG_ERROR("[%s]input tensor alloc_qnn_tensor_id failed", _name.c_str());
            return false;
        }

        QNN_LOG_DEBUG("[%s]input tensor id: %d", _name.c_str(), tensor->get_qnn_tensor_id());
        _qnn_tensor_inputs[i] = tensor->get_qnn_tensor();
    }

    for (size_t i = 0; i < _tensor_outputs.size(); i++) {
        auto tensor = _tensor_outputs[i];
        if (!tensor->alloc_qnn_tensor_id()) {
            QNN_LOG_ERROR("[%s]output tensor alloc_qnn_tensor_id failed", _name.c_str());
            return false;
        }

        QNN_LOG_DEBUG("[%s]output tensor id: %d", _name.c_str(), tensor->get_qnn_tensor_id());
        _qnn_tensor_outputs[i] = tensor->get_qnn_tensor();
    }

    auto qnn_interface = _qnn_instance->get_qnn_interface();
    auto error = qnn_interface->qnn_graph_add_node(graph_handle, get_op_config());
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s]qnn_graph_add_node.error: %s", _name.c_str(), get_qnn_error_string(error));
        return false;
    }

    QNN_LOG_DEBUG("[%s]added to graph succeed", _name.c_str());
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

bool ggml_qnn_single_op_config::initialize_op_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
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

bool ggml_qnn_aggregate_op_config::bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) {
    return bind_tensors(tensor_inputs, _tensor_inputs, _qnn_tensor_inputs);
}

bool ggml_qnn_aggregate_op_config::bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) {
    return bind_tensors(tensor_outputs, _tensor_outputs, _qnn_tensor_outputs);
}

bool ggml_qnn_matmul_op_config::initialize_op_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
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
    params.name_prefix = "dst";
    params.is_input = false;
    create_tensors_from_ggml_tensor(params, tensor_outputs, &_tensor_outputs, &_qnn_tensor_outputs);

    // create convert nodes
    qnn_tensor_array_t mat_mul_tensor_inputs = _tensor_inputs;
    qnn_tensor_array_t mat_mul_tensor_outputs = _tensor_outputs;
    if (!create_convert_nodes(device, graph_handle, tensor_rank, mat_mul_tensor_inputs, mat_mul_tensor_outputs)) {
        QNN_LOG_ERROR("create convert nodes failed");
        return false;
    }

    mat_mul_tensor_inputs.front() =
        create_gather_nodes(device, graph_handle, tensor_rank, mat_mul_tensor_inputs.front(),
                            mat_mul_tensor_inputs.back()->get_dimensions());
    return create_mat_mul_nodes(device, graph_handle, tensor_rank, mat_mul_tensor_inputs, mat_mul_tensor_outputs);
}

qnn_tensor_ptr_t ggml_qnn_matmul_op_config::create_gather_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                                                const int rank, qnn_tensor_ptr_t tensor_input,
                                                                qnn_dimension_array_t output_dimensions) {
    if (rank <= 2) {
        return tensor_input;
    }

    const auto &input_dimensions = tensor_input->get_dimensions();
    output_dimensions[rank - 1] = input_dimensions[rank - 1];
    output_dimensions[rank - 2] = input_dimensions[rank - 2];

    const auto y = output_dimensions[rank - 3] / input_dimensions[rank - 3];
    if (y == 1 && (rank == 3 || (rank == 4 && output_dimensions[rank - 4] == input_dimensions[rank - 4]))) {
        return tensor_input;
    }

    // create concat nodes, to convert tensor shape from [ne03, ne02, n, k] to [ne03 * x, ne02 * y, n, k]
    constexpr const auto create_node =
        [](const std::string &name, const int rank, const int axis, const qnn_dimension_array_t &dimensions,
           qnn_tensor_ptr_t tensor_input, QNNBackend device, Qnn_GraphHandle_t graph_handle,
           std::shared_ptr<qnn_instance> qnn_instance, qnn_tensor_ptr_t &tensor_output) -> qnn_op_config_ptr_t {
        auto gather_out =
            std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, name + "_out", dimensions,
                                              tensor_input->get_data_type(), rank, device, graph_handle, qnn_instance);
        auto gather_op = std::make_shared<ggml_qnn_connectable_op_config>(name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                          QNN_OP_GATHER, qnn_instance);

        Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
        scalar.dataType = QNN_DATATYPE_INT_32;
        scalar.int32Value = axis;
        gather_op->add_scalar_param(QNN_OP_GATHER_PARAM_AXIS, scalar);
        gather_op->set_output_tensors({gather_out});

        // here we calculate the index mapping, will generate a 1d tensor like [0, 0, 0, 1, 1, 1, 2, 2, 2, ...],
        //   by repeating each index [scale] times.
        const auto scale = dimensions[axis] / tensor_input->get_dimensions()[axis];
        std::vector<uint8_t> index_buffer(dimensions[axis] * sizeof(uint32_t));
        for (uint32_t *curr = reinterpret_cast<uint32_t *>(index_buffer.data()), *end = curr + dimensions[axis];
             curr < end; curr++) {
            *curr = (curr - reinterpret_cast<uint32_t *>(index_buffer.data())) / scale;
        }

        auto gather_index = std::make_shared<ggml_qnn_tensor>(
            ggml_qnn_tensor::PARAMETER, name + "_index", qnn_dimension_array_t{dimensions[axis]}, QNN_DATATYPE_UINT_32,
            1, device, graph_handle, qnn_instance);
        gather_index->set_data_buffer(std::move(index_buffer));
        gather_op->set_input_tensors({tensor_input, gather_index});

        tensor_output = gather_out;
        return gather_op;
    };

    qnn_dimension_array_t intermediate_dimensions = input_dimensions;
    intermediate_dimensions[rank - 3] = output_dimensions[rank - 3];
    qnn_tensor_ptr_t gather0_out;
    _operations.push_back(create_node(_name + "_gather0", rank, rank - 3, intermediate_dimensions, tensor_input, device,
                                      graph_handle, _qnn_instance, gather0_out));
    if (rank == 3) {
        return gather0_out;
    }

    qnn_tensor_ptr_t gather1_out;
    _operations.push_back(create_node(_name + "_gather1", rank, rank - 4, output_dimensions, gather0_out, device,
                                      graph_handle, _qnn_instance, gather1_out));
    return gather1_out;
}

bool ggml_qnn_matmul_op_config::create_convert_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                                                     qnn_tensor_array_t &tensor_inputs,
                                                     qnn_tensor_array_t &tensor_outputs) {
    if (device == QNN_BACKEND_GPU) {
        // there's no convert op for GPU, so we should create matmul nodes directly.
        return true;
    }

    // create tensors for convert node
    auto tensor_type = get_tensor_type(tensor_inputs);
    QNN_LOG_DEBUG("input tensor type: %s", qnn_datatype_to_string(tensor_type));

    for (size_t i = 0; i < tensor_inputs.size(); ++i) {
        // create input convert nodes
        auto convert_in = tensor_inputs[i];
        if (convert_in->get_data_type() == tensor_type) {
            continue;
        }

        std::string convert_name("convert_src" + std::to_string(i));
        auto convert_out = std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, convert_name + "_out",
                                                             convert_in->get_dimensions(), tensor_type, rank, device,
                                                             graph_handle, _qnn_instance);
        auto convert = std::make_shared<ggml_qnn_connectable_op_config>(convert_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                        QNN_OP_CONVERT, _qnn_instance);
        convert->set_input_tensors({convert_in});
        convert->set_output_tensors({convert_out});
        tensor_inputs[i] = convert_out;
        _operations.push_back(convert);
    }

    if (tensor_outputs.front()->get_data_type() != tensor_type) {
        // create output convert node
        std::string convert_name("convert_dst");
        auto convert_out = tensor_outputs.front();
        auto convert_in = std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, convert_name + "_in",
                                                            convert_out->get_dimensions(), tensor_type, rank, device,
                                                            graph_handle, _qnn_instance);
        auto output_convert = std::make_shared<ggml_qnn_connectable_op_config>(
            convert_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONVERT, _qnn_instance);
        output_convert->set_input_tensors({convert_in});
        output_convert->set_output_tensors({convert_out});
        tensor_outputs.front() = convert_in;
        _operations.push_back(output_convert);
    }

    return true;
}

bool ggml_qnn_matmul_op_config::create_mat_mul_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle, const int rank,
                                                     qnn_tensor_array_t &tensor_inputs,
                                                     qnn_tensor_array_t &tensor_outputs) {

    /*
     * First, both the ggml and qnn tensor in memory are stored as row-major format. (For more details, please refer to:
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
     *        i2>ggml_tensor_in1] --src1--> mat_mul0;
     *        mat_mul0 --dst_trans--> transpose_out;
     *        transpose1 --dst0--> o1>ggml_tensor_out];
     *   ```
     */

    // create src0_trans tensor
    auto src1 = tensor_inputs.back();
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS does not match the expected value");

    qnn_dimension_array_t dimensions = get_transposed_dimensions(src1->get_dimensions(), rank);

    // create dst_trans tensor
    auto dst = tensor_outputs.front();
    dimensions = get_transposed_dimensions(dst->get_dimensions(), rank);
    auto dst_trans = std::make_shared<ggml_qnn_tensor>(ggml_qnn_tensor::INTERMEDIATE, "dst_trans", dimensions,
                                                       dst->get_data_type(), rank, device, graph_handle, _qnn_instance);

    // create transpose_out
    auto transpose_out = std::make_shared<ggml_qnn_connectable_op_config>(
        _name + "_trans1", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, _qnn_instance);

    // create mat_mul
    auto mat_mul = std::make_shared<ggml_qnn_connectable_op_config>(_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                                                    _qnn_instance);

    Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
    scalar.dataType = QNN_DATATYPE_BOOL_8;
    scalar.bool8Value = 1;
    mat_mul->add_scalar_param(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar);

    // set transpose_out parameters
    auto *params_data = reinterpret_cast<const uint8_t *>(kTransposeParamData[rank - 1].data());
    const qnn_dimension_array_t param_dims = {(uint32_t)rank, 1, 1, 1};
    transpose_out->add_tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, param_dims, 1, params_data, QNN_DATATYPE_UINT_32,
                                    device, graph_handle);

    // set tensor to mat_mul
    mat_mul->set_input_tensors(tensor_inputs);
    qnn_tensor_array_t tensors = {dst_trans};
    mat_mul->set_output_tensors(tensors);

    // set tensor to transpose_out
    tensors = {dst_trans};
    transpose_out->set_input_tensors(tensors);
    transpose_out->set_output_tensors(tensor_outputs);

    _operations.push_back(mat_mul);
    _operations.push_back(transpose_out);
    return true;
}

ggml_op_constructor_t create_op_constructor(const std::string &op_name) {
    if (op_name == QNN_OP_MAT_MUL) {
        // For QNN_OP_MAT_MUL, we need to transpose the input tensor
        return [](const std::string &instance_name,
                  std::shared_ptr<qnn::qnn_instance> qnn_instance) -> std::unique_ptr<qnn::ggml_qnn_op_config> {
            QNN_LOG_DEBUG("create QNN_OP_MAT_MUL, name %s", instance_name.c_str());
            return std::make_unique<qnn::ggml_qnn_matmul_op_config>(instance_name, qnn_instance);
        };
    }

    return [op_name](const std::string &instance_name,
                     std::shared_ptr<qnn::qnn_instance> qnn_instance) -> std::unique_ptr<qnn::ggml_qnn_op_config> {
        return std::make_unique<qnn::ggml_qnn_single_op_config>(instance_name, QNN_OP_PACKAGE_NAME_QTI_AISW, op_name,
                                                                qnn_instance);
    };
}

} // namespace qnn
