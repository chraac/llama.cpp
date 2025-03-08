
#include "graph.hpp"

#include <algorithm>
#include <unordered_map>

#include "ggml-impl.h"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"

namespace {
using qnn_tensor_cache_t = std::unordered_map<ggml_tensor *, qnn::qnn_tensor_ptr_t>;

int get_op_max_rank(const ggml_tensor * op) {
    int max_rank = ggml_n_dims(op);
    for (int i = 0; i < GGML_MAX_DIMS && op->src[i]; ++i) {
        max_rank = std::max(max_rank, ggml_n_dims(op->src[i]));
    }

    return max_rank;
}

qnn::qnn_tensor_ptr_t create_tensor_with_cache(ggml_tensor * tensor, qnn::ggml_qnn_tensor::tensor_type_t type, int rank,
                                               ggml_type override_data_type, QNNBackend device,
                                               Qnn_GraphHandle_t                  graph_handle,
                                               std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                               qnn_tensor_cache_t &               tensor_cache) {
    GGML_ASSERT(tensor);
    if (tensor_cache.count(tensor)) {
        return tensor_cache[tensor];
    }

    QNN_LOG_DEBUG("[%s]create_tensor_with_cache, data_type: %s, override_data_type: %s\n",
                  qnn::get_backend_name(device), ggml_type_name(tensor->type), ggml_type_name(override_data_type));
    auto data_type  = override_data_type != GGML_TYPE_COUNT ? override_data_type : tensor->type;
    auto qnn_tensor = std::make_shared<qnn::ggml_qnn_tensor>(type, tensor->name, tensor->ne, data_type, rank, device,
                                                             graph_handle, qnn_instance);
    tensor_cache[tensor] = qnn_tensor;
    return qnn_tensor;
}

qnn::qnn_tensor_array_t create_tensors_with_cache(const qnn::ggml_tensor_array_t &    ggml_tensors,
                                                  qnn::ggml_qnn_tensor::tensor_type_t type, int rank,
                                                  ggml_type override_data_type, QNNBackend device,
                                                  Qnn_GraphHandle_t                  graph_handle,
                                                  std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                  qnn_tensor_cache_t &               tensor_cache) {
    qnn::qnn_tensor_array_t tensors;
    for (auto * tensor : ggml_tensors) {
        tensors.push_back(create_tensor_with_cache(tensor, type, rank, override_data_type, device, graph_handle,
                                                   qnn_instance, tensor_cache));
    }

    return tensors;
}

qnn::qnn_op_config_ptr_t create_operation_from_op_tensor(ggml_tensor * dst, const std::string & name, int rank,
                                                         QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                                         std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                         qnn_tensor_cache_t &               tensor_cache) {
    auto operation = qnn::create_op(dst, name, qnn_instance);

    // input tensors
    qnn::qnn_tensor_array_t input_qnn_tensors;
    for (size_t i = 0; i < GGML_MAX_DIMS && dst->src[i]; ++i) {
        auto * src            = dst->src[i];
        auto input_qnn_tensor = create_tensor_with_cache(src, qnn::ggml_qnn_tensor::INTERMEDIATE, rank, GGML_TYPE_COUNT,
                                                         device, graph_handle, qnn_instance, tensor_cache);
        input_qnn_tensors.push_back(input_qnn_tensor);
    }
    operation->set_input_tensors(input_qnn_tensors);

    // output tensor
    qnn::qnn_tensor_array_t output_qnn_tensors =
        create_tensors_with_cache({ dst }, qnn::ggml_qnn_tensor::INTERMEDIATE, rank, GGML_TYPE_COUNT, device,
                                  graph_handle, qnn_instance, tensor_cache);
    operation->set_output_tensors(output_qnn_tensors);

    // initialize operation
    if (!operation->initialize_op_nodes(device, graph_handle)) {
        QNN_LOG_ERROR("[%s][%s]initialize_op_nodes failed\n", qnn::get_backend_name(device), name.c_str());
        return nullptr;
    }

    return operation;
}

/**
 * @brief Extracts input and output tensors from a computational graph.
 *
 * This function identifies the input and output tensors of a computational graph by analyzing the connectivity between
 * tensor nodes. It does this by iterating over each node in the graph, using a connectivity map that associates every
 * tensor with its number of incoming connections (in_degree), outgoing connections (out_degree), and an insertion index
 * that preserves order. The insertion index is used later to sort the tensors in their original discovery order.
 *
 * TODO: this algorithm is not perfect and may not work for all cases. It assumes that the tensors are
 *   connected in a way that allows for unambiguous categorization.
 */
int get_io_tensors_from_graph(const ggml_cgraph * cgraph, qnn::ggml_tensor_array_t & inputs,
                              qnn::ggml_tensor_array_t & outputs) {
    struct _tensor_connectivity_info {
        size_t in_degree    = 0;
        size_t out_degree   = 0;
        size_t insert_index = 0;
    };

    using ggml_tensor_connectivity_map_t = std::unordered_map<ggml_tensor *, _tensor_connectivity_info>;

    ggml_tensor_connectivity_map_t connectivity_map;
    int                            rank = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * dst = cgraph->nodes[i];
        if (ggml_is_empty(dst)) {
            continue;
        }

        if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW || dst->op == GGML_OP_PERMUTE) {
            // TODO: remove GGML_OP_VIEW after view op is supported
            continue;
        }

        rank = std::max(rank, ggml_n_dims(dst));
        if (connectivity_map.count(dst) == 0) {
            connectivity_map[dst] = {
                1,  // in-degree, at least 1
                0,
                connectivity_map.size(),
            };
        } else {
            ++(connectivity_map[dst].in_degree);
        }

        for (size_t i = 0; i < GGML_MAX_DIMS && dst->src[i]; ++i) {
            auto * src = dst->src[i];
            rank       = std::max(rank, ggml_n_dims(src));

            if (connectivity_map.count(src) == 0) {
                connectivity_map[src] = {
                    0,
                    1,  // out-degree, at least 1
                    connectivity_map.size(),
                };
            } else {
                ++(connectivity_map[src].out_degree);
            }
        }
    }

    for (const auto & kv : connectivity_map) {
        if (kv.second.in_degree == 0) {
            inputs.push_back(kv.first);
        }

        if (kv.second.out_degree == 0) {
            outputs.push_back(kv.first);
        }
    }

    std::sort(inputs.begin(), inputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    std::sort(outputs.begin(), outputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    return rank;
}

/*
 * for src0_F32, src1_F32, dst_F32 -> GGML_TYPE_COUNT
 * for src0_F16, src1_F16, dst_F16 -> GGML_TYPE_COUNT
 * for src0_F16, src1_F32, dst_F32 -> GGML_TYPE_F32
 * for src0_q4, src1_F32, dst_F32 -> GGML_TYPE_F32
 * for src0_q4, src1_F16, dst_F32 -> GGML_TYPE_F32
 */
ggml_type get_override_data_type(const qnn::ggml_tensor_array_t & inputs, const qnn::ggml_tensor_array_t & outputs) {
    GGML_ASSERT(!inputs.empty());
    ggml_type override_data_type = inputs.front()->type;
    bool      is_same_data_type  = true;
    for (auto * tensor : inputs) {
        is_same_data_type  = is_same_data_type && tensor->type == override_data_type;
        override_data_type = std::min(override_data_type, tensor->type);
    }

    for (auto * tensor : outputs) {
        is_same_data_type  = is_same_data_type && tensor->type == override_data_type;
        override_data_type = std::min(override_data_type, tensor->type);
    }

    return is_same_data_type ? GGML_TYPE_COUNT : override_data_type;
}

static const QnnHtpGraph_CustomConfig_t kDefaultHvxConfig = []() {
    QnnHtpGraph_CustomConfig_t hvx_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    hvx_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    hvx_config.numHvxThreads              = 8;
    return hvx_config;
}();

static const QnnHtpGraph_CustomConfig_t kDefaultDlbcConfig = []() {
    QnnHtpGraph_CustomConfig_t dlbc_config    = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    dlbc_config.option                        = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    dlbc_config.optimizationOption.type       = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
    dlbc_config.optimizationOption.floatValue = 1.0;  // set to 0.0 to turn off DLBC
    return dlbc_config;
}();

/*
 *  1 = Faster preparation time, less optimal graph
 *  2 = Longer preparation time, more optimal graph
 *  3 = Longest preparation time, most likely even more optimal graph:
 *   QNN_HTP_DEVICE_CONFIG_OPTION_SOC configuration will be taken into account when possible, details see HTP Backend Specific Page
 */
static const QnnHtpGraph_CustomConfig_t kDefaultOptConfig = []() {
    QnnHtpGraph_CustomConfig_t opt_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    opt_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    opt_config.optimizationOption.type    = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
#ifndef NDEBUG
    opt_config.optimizationOption.floatValue = 3;
#else
    opt_config.optimizationOption.floatValue = 1;
#endif
    return opt_config;
}();

static const QnnHtpGraph_CustomConfig_t kHtpPrecisionConfigF16 = []() {
    QnnHtpGraph_CustomConfig_t precision_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    precision_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    precision_config.precision                  = QNN_PRECISION_FLOAT16;
    return precision_config;
}();

constexpr QnnHtpGraph_CustomConfig_t make_vtcm_config(size_t vtcm_size_in_mb) {
    QnnHtpGraph_CustomConfig_t vtcm_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    vtcm_config.option                     = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
    vtcm_config.vtcmSizeInMB               = (uint32_t) vtcm_size_in_mb;
    return vtcm_config;
}

constexpr QnnGraph_Config_t make_graph_config(const QnnHtpGraph_CustomConfig_t * custom_config) {
    QnnGraph_Config_t graph_config = QNN_GRAPH_CONFIG_INIT;
    graph_config.option            = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_config.customConfig      = const_cast<QnnHtpGraph_CustomConfig_t *>(custom_config);
    return graph_config;
}

}  // namespace

namespace qnn {

qnn_graph::qnn_graph(const std::string & graph_name, QNNBackend device, std::shared_ptr<qnn_instance> qnn_instance,
                     htp_precision precision, size_t vtcm_size_in_mb) :
    _graph_name(graph_name),
    _device(device),
    _qnn_instance(qnn_instance) {
    QNN_LOG_DEBUG("[%s][%s]created\n", get_backend_name(device), graph_name.c_str());

    auto              qnn_interface = qnn_instance->get_qnn_interface();
    auto              qnn_context   = qnn_instance->get_qnn_context_handle();
    Qnn_ErrorHandle_t error         = QNN_SUCCESS;
    Qnn_GraphHandle_t graph_handle  = nullptr;
    if (device == QNN_BACKEND_NPU) {
        // TODO: fix graph config here for NPU
        std::vector<const QnnGraph_Config_t *> graph_configs;

        auto hvx_config = make_graph_config(&kDefaultHvxConfig);
        graph_configs.push_back(&hvx_config);

        auto dlbc_config = make_graph_config(&kDefaultDlbcConfig);
        graph_configs.push_back(&dlbc_config);

        auto opt_config = make_graph_config(&kDefaultOptConfig);
        graph_configs.push_back(&opt_config);

        auto vctm_sub_config = make_vtcm_config(vtcm_size_in_mb);
        auto vtcm_config     = make_graph_config(&vctm_sub_config);
        graph_configs.push_back(&vtcm_config);

        if (precision == qnn_graph::kHtpFp16) {
            auto precision_config = make_graph_config(&kHtpPrecisionConfigF16);
            graph_configs.push_back(&precision_config);
        }

        graph_configs.push_back(nullptr);
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), graph_configs.data(), &graph_handle);
    } else {
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), nullptr, &graph_handle);
    }

    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s\n", get_backend_name(device), graph_name.c_str(),
                      get_qnn_error_string(error));
        return;
    }

    _graph_handle  = graph_handle;
    _qnn_interface = qnn_interface;
    QNN_LOG_DEBUG("[%s][%s]create succeed\n", get_backend_name(device), graph_name.c_str());
}

qnn_graph::~qnn_graph() {
    QNN_LOG_DEBUG("[%s][%s]destroy\n", get_backend_name(_device), _graph_name.c_str());
}

bool qnn_graph::build_graph_from_ggml_graph(const ggml_cgraph * cgraph) {
    QNN_LOG_DEBUG("[%s][%s]build start\n", get_backend_name(_device), _graph_name.c_str());

    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
    int                 rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s][%s]rank: %d, input_set: %d, output_set: %d\n", get_backend_name(_device), _graph_name.c_str(),
                  rank, int(inputs.size()), int(outputs.size()));

    {
        static_assert(
            GGML_TYPE_COUNT > GGML_TYPE_Q8_0 && GGML_TYPE_Q8_0 > GGML_TYPE_F16 && GGML_TYPE_F16 > GGML_TYPE_F32,
            "GGML_TYPE enum order is not correct");

        _override_data_type = get_override_data_type(inputs, outputs);
        if (_override_data_type != GGML_TYPE_COUNT) {
            QNN_LOG_DEBUG("[%s][%s]set override_data_type: %s\n", get_backend_name(_device), _graph_name.c_str(),
                          ggml_type_name(_override_data_type));
        }

        qnn_tensor_cache_t tensor_cache;
        auto input_tensors  = create_tensors_with_cache(inputs, ggml_qnn_tensor::INPUT, rank, _override_data_type,
                                                        _device, _graph_handle, _qnn_instance, tensor_cache);
        auto output_tensors = create_tensors_with_cache(outputs, ggml_qnn_tensor::OUTPUT, rank, GGML_TYPE_COUNT,
                                                        _device, _graph_handle, _qnn_instance, tensor_cache);
        qnn_op_config_array_t operations;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * dst = cgraph->nodes[i];
            if (ggml_is_empty(dst)) {
                continue;
            }

            if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW || dst->op == GGML_OP_PERMUTE) {
                // TODO: remove GGML_OP_VIEW after view op is supported
                continue;
            }

            QNN_LOG_DEBUG("[%s]create op: %s\n", get_backend_name(_device), get_qnn_op_name(dst));
            auto operation = create_operation_from_op_tensor(dst, dst->name, rank, _device, _graph_handle,
                                                             _qnn_instance, tensor_cache);  // TODO: fix op name
            operations.push_back(operation);
        }

        _tensor_inputs  = std::move(input_tensors);
        _tensor_outputs = std::move(output_tensors);
        _operations     = std::move(operations);
        if (!finalize()) {
            return false;
        }
    }

    QNN_LOG_DEBUG("[%s][%s]build succeed\n", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::execute(const ggml_cgraph * cgraph, std::shared_ptr<qnn_convert_context_t> convert_context) {
    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
#ifdef NDEBUG
    get_io_tensors_from_graph(cgraph, inputs, outputs);
#else
    int rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s]rank: %d, input_set: %d, output_set: %d\n", get_backend_name(_device), rank, int(inputs.size()),
                  int(outputs.size()));
#endif

    {
        if (_override_data_type != GGML_TYPE_COUNT) {
            QNN_LOG_DEBUG("[%s][%s]override_data_type: %s\n", get_backend_name(_device), _graph_name.c_str(),
                          ggml_type_name(_override_data_type));
            auto buffers = convert(convert_context, inputs, _override_data_type);
            if (!qnn::bind_tensors_with_custom_buffers(inputs, buffers, _tensor_inputs, _qnn_tensor_inputs)) {
                QNN_LOG_ERROR("[%s][%s]bind input tensors failed\n", get_backend_name(_device), _graph_name.c_str());
                return false;
            }
        } else {
            if (!qnn::bind_tensors(inputs, _tensor_inputs, _qnn_tensor_inputs)) {
                QNN_LOG_ERROR("[%s][%s]bind input tensors failed\n", get_backend_name(_device), _graph_name.c_str());
                return false;
            }
        }

        if (!qnn::bind_tensors(outputs, _tensor_outputs, _qnn_tensor_outputs)) {
            QNN_LOG_ERROR("[%s][%s]bind output tensors failed\n", get_backend_name(_device), _graph_name.c_str());
            return false;
        }

        auto & qnn_tensor_inputs  = _qnn_tensor_inputs;
        auto & qnn_tensor_outputs = _qnn_tensor_outputs;
        auto   error =
            _qnn_interface->qnn_graph_execute(_graph_handle, qnn_tensor_inputs.data(), qnn_tensor_inputs.size(),
                                              qnn_tensor_outputs.data(), qnn_tensor_outputs.size(), nullptr, nullptr);
        unbind_tensors(_tensor_inputs);
        unbind_tensors(_tensor_outputs);

        if (error != QNN_SUCCESS) {
            if (_device == QNN_BACKEND_NPU && error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION) {
                QNN_LOG_WARN("[%s][%s][graph_execute]NPU crashed. SSR detected. Caused QNN graph execute error.\n",
                             get_backend_name(_device), _graph_name.c_str());
            } else {
                QNN_LOG_ERROR("[%s][%s][graph_execute]error: %s\n", get_backend_name(_device), _graph_name.c_str(),
                              get_qnn_error_string(error));
            }
            return false;
        }

        QNN_LOG_DEBUG("[%s][%s]execute succeed\n", get_backend_name(_device), _graph_name.c_str());
        return true;
    }
}

bool qnn_graph::finalize() {
    if (!qnn::add_op_to_graph(_graph_handle, _operations)) {
        QNN_LOG_ERROR("[%s]add nodes failed\n", _graph_name.c_str());
        return false;
    }

    auto error = _qnn_interface->qnn_graph_finalize(_graph_handle, nullptr, nullptr);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]qnn_graph_finalize.error: %s\n", get_backend_name(_device), _graph_name.c_str(),
                      get_qnn_error_string(error));
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]finalize succeed\n", get_backend_name(_device), _graph_name.c_str());
    return true;
}

}  // namespace qnn
