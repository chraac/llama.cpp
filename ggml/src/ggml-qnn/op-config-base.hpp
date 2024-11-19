#pragma once

#include <memory>
#include <vector>

#include "ggml-qnn.h"

#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {

using ggml_tensor_array_t = std::vector<ggml_tensor *>;

/**
 * @class ggml_qnn_op_config
 * @brief Abstract base class for configuring QNN operations.
 *
 * This class provides an interface for creating and managing tensors,
 * adding operations to a graph, and binding/unbinding input and output tensors.
 */
class ggml_qnn_op_config {
public:
    virtual ~ggml_qnn_op_config() {}

    /**
     * @brief Creates tensors and internal nodes for constructing the calculation graph.
     *
     * This pure virtual function is responsible for creating tensors on the given
     * backend device, associating them with the provided graph handle, and creating
     * the internal nodes necessary for constructing the calculation graph. It takes
     * input and output tensor arrays as parameters.
     *
     * @param device The backend device where tensors will be created.
     * @param graph_handle The handle to the graph where tensors and nodes will be associated.
     * @param tensor_inputs An array of input tensors.
     * @param tensor_outputs An array of output tensors.
     * @return true if tensors and nodes are successfully created, false otherwise.
     */
    virtual bool initialize_op_nodes(QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                     const ggml_tensor_array_t &tensor_inputs,
                                     const ggml_tensor_array_t &tensor_outputs) = 0;

    /**
     * @brief Pure virtual function to retrieve the input tensors for QNN (Quantized Neural Network).
     *
     * This function must be overridden by derived classes to provide the specific implementation
     * for retrieving the input tensors used in QNN operations.
     *
     * @return A reference to a vector of Qnn_Tensor_t objects representing the input tensors.
     */
    virtual std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() = 0;

    /**
     * @brief Pure virtual function to retrieve the output tensors of a QNN (Quantized Neural Network).
     *
     * This function must be overridden by any derived class to provide access to the
     * output tensors of the QNN. The function returns a reference to a vector of
     * Qnn_Tensor_t objects, which represent the output tensors.
     *
     * @return std::vector<Qnn_Tensor_t>& Reference to a vector of Qnn_Tensor_t objects.
     */
    virtual std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() = 0;

    /**
     * @brief Adds an operation to the given graph.
     *
     * This pure virtual function must be implemented by derived classes to add
     * a specific operation to the provided graph handle.
     *
     * This function will be called after `initialize_op_nodes` during initialization.
     *
     * @param graph_handle The handle to the graph where the operation will be added.
     * @return true if the operation was successfully added to the graph, false otherwise.
     */
    virtual bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) = 0;

    /**
     * @brief Binds the input tensors to the operation.
     *
     * This pure virtual function must be implemented by derived classes to bind
     * the provided input tensors to the operation. The function takes a constant
     * reference to a ggml_tensor_array_t object, which contains the input tensors
     * to be bound.
     *
     * @param tensor_inputs A constant reference to a ggml_tensor_array_t object
     *                      containing the input tensors.
     * @return true if the input tensors were successfully bound, false otherwise.
     */
    virtual bool bind_input_tensors(const ggml_tensor_array_t &tensor_inputs) = 0;

    /**
     * @brief Binds the output tensors to the given tensor array.
     *
     * This pure virtual function must be implemented by derived classes to bind
     * the output tensors to the provided array of tensors. The function is expected
     * to establish the necessary connections or mappings between the output tensors
     * and the elements of the given tensor array.
     *
     * @param tensor_outputs A constant reference to an array of ggml tensors that
     *                       represent the output tensors to be bound.
     * @return true if the binding is successful, false otherwise.
     */
    virtual bool bind_output_tensors(const ggml_tensor_array_t &tensor_outputs) = 0;

    /**
     * @brief Unbinds the input tensors from the operation.
     *
     * This pure virtual function is intended to be overridden by derived classes
     * to implement the logic for unbinding or detaching input tensors that were
     * previously bound to the operation. This is typically used to release resources
     * or reset the state of the operation.
     */
    virtual void unbind_input_tensors() = 0;

    /**
     * @brief Unbinds the output tensors.
     *
     * This pure virtual function is responsible for unbinding or detaching
     * the output tensors from their current bindings. Implementations of this
     * function should ensure that any resources or references held by the
     * output tensors are properly released or reset.
     */
    virtual void unbind_output_tensors() = 0;
};

using qnn_op_config_ptr_t = std::shared_ptr<ggml_qnn_op_config>;

} // namespace qnn
