
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "convert.hpp"
#include "ggml-qnn.h"
#include "op-config.hpp"
#include "qnn-lib.hpp"

namespace qnn {

/**
 * @class qnn_graph
 * @brief Manages a QNN graph, converting a GGML graph to QNN format and handling its execution.
 *
 * This class is responsible for building a QNN graph from a given GGML graph,
 * determining its input/output tensors, finalizing the configuration, and
 * executing the graph on the specified backend device.
 */
class qnn_graph {
  public:
    enum htp_precision {
        kHtpDefault = 0,
        kHtpFp16,
    };

    explicit qnn_graph(const std::string & graph_name, QNNBackend device, qnn_instance_ptr qnn_instance,
                       htp_precision precision, size_t vtcm_size_in_mb);

    ~qnn_graph();

    bool build_graph_from_ggml_graph(const ggml_cgraph * cgraph);

    bool execute(const ggml_cgraph * cgraph, std::shared_ptr<qnn_convert_context_t> convert_context);

    bool is_valid() const { return _graph_handle != nullptr; }

    Qnn_GraphHandle_t get_graph_handler() const { return _graph_handle; }

    qnn_instance_ptr get_qnn_instance() { return _qnn_instance; }

    const std::string & get_name() const { return _graph_name; }

    QNNBackend get_device() const { return _device; }

  private:
    bool finalize();

    const std::string     _graph_name;
    const QNNBackend      _device;
    Qnn_GraphHandle_t     _graph_handle = nullptr;
    qnn_instance_ptr      _qnn_instance;
    qnn_interface_ptr     _qnn_interface;
    qnn_op_config_array_t _operations;

    qnn_tensor_array_t        _tensor_inputs;
    qnn_tensor_array_t        _tensor_outputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;

    // the actual data type to use for the input tensors of qnn graph
    ggml_type _override_data_type = GGML_TYPE_COUNT;

    DISABLE_COPY(qnn_graph);
    DISABLE_MOVE(qnn_graph);
};

using qnn_graph_ptr_t = std::shared_ptr<qnn_graph>;

}  // namespace qnn
