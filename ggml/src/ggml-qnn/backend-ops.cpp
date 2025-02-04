
#include "backend-ops.hpp"

#include <memory>

#include "ggml-impl.h"

#include "graph.hpp"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

void append_tensor_dimensions(const ggml_tensor *tensor, std::string &output) {
    char buffer[256] = {};
    const auto *type_name = qnn::get_ggml_type_name(tensor->type);
    int len = 0;
    switch (ggml_n_dims(tensor)) {
        case 1:
            len = snprintf(buffer, sizeof(buffer), "%ld%s", (long)tensor->ne[0], type_name);
            break;
        case 2:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1], type_name);
            break;
        case 3:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], type_name);
            break;
        case 4:
        default:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], (long)tensor->ne[3], type_name);
            break;
    }
    GGML_ASSERT(len > 0 && len < (int)sizeof(buffer));
    output.append(buffer, len);
}

void get_graph_key_from_op(const ggml_tensor *op, std::string &output) {
    GGML_ASSERT(op->op != GGML_OP_NONE);
    output += ggml_op_desc(op);
    output += qnn::get_ggml_type_name(op->type);
    const auto param_count = qnn::get_qnn_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        auto *input = op->src[i];
        if (!input) {
            break;
        }

        output += '_';
        append_tensor_dimensions(input, output);
    }
}

void get_op_key_with_src_op_desc(const ggml_tensor *op, std::string &output) {
    output += ggml_op_desc(op);
    output += '(';
    if (op->src[0]) {
        output += ggml_op_desc(op->src[0]);
    }
    for (size_t i = 1; i < GGML_MAX_DIMS && op->src[i]; ++i) {
        output += ',';
        output += ggml_op_desc(op->src[i]);
    }
    output += ')';
}

void get_graph_key_from_cgraph(const ggml_cgraph *cgraph, std::string &output) {
    // generate key from the graph, the key is used to cache the graph, like:
    //   "MUL_MATf32_256x16x10f32_256x1x10f32#LOG#ADD#ADDf32_16x1x10f32"
    if (cgraph->n_nodes == 0) {
        QNN_LOG_DEBUG("empty cgraph");
        return;
    }

    {
        bool is_start = true;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            auto *op = cgraph->nodes[i];
            if (ggml_is_empty(op)) {
                QNN_LOG_DEBUG("empty op in graph, skipping");
                continue;
            }

            if (op->op == GGML_OP_NONE) {
                QNN_LOG_DEBUG("GGML_OP_NONE in graph, skipping");
                continue;
            }

            if (is_start) {
                get_graph_key_from_op(cgraph->nodes[0], output);
                is_start = false;
            } else {
                output += '#';
                get_op_key_with_src_op_desc(op, output);
            }
        }
    }

    if (cgraph->n_nodes > 1) {
        auto *last_op = cgraph->nodes[cgraph->n_nodes - 1];
        output += qnn::get_ggml_type_name(last_op->type);
        output += '_';
        append_tensor_dimensions(last_op, output);
    }
}

qnn::qnn_graph *get_qnn_graph_from_cache(ggml_backend_qnn_device_context *ctx, const ggml_cgraph *cgraph) {
    auto &graph_cache = ctx->qnn_graph_cache;
    std::string graph_key;
    get_graph_key_from_cgraph(cgraph, graph_key);
    if (graph_key.empty()) {
        QNN_LOG_DEBUG("[%s]empty graph key for cgraph: %p, size: %d", qnn::get_backend_name(ctx->device), cgraph,
                      (int)cgraph->n_nodes);
        return nullptr;
    }

    auto it = graph_cache.find(graph_key);
    qnn::qnn_graph *graph_ptr = nullptr;
    if (it != graph_cache.end()) {
        QNN_LOG_DEBUG("[%s]found graph %s in cache", qnn::get_backend_name(ctx->device), graph_key.c_str());
        graph_ptr = it->second.get();
    } else {
        auto graph =
            std::make_unique<qnn::qnn_graph>(graph_key, ctx->device, ctx->instance, ctx->socinfo.vtcm_size_in_mb);
        if (!graph->is_valid()) {
            return nullptr;
        }

        if (!graph->build_graph_from_ggml_graph(cgraph)) {
            QNN_LOG_ERROR("[%s]build_graph_from_ggml_graph failed", qnn::get_backend_name(ctx->device));
            return nullptr;
        }

        graph_ptr = graph.get();
        graph_cache[graph_key] = std::move(graph);
    }

    return graph_ptr;
}

constexpr bool isQnnSupportedOp(size_t op_index) {
    switch (op_index) {
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_MUL_MAT:
        case GGML_OP_RESHAPE:
        case (GGML_OP_COUNT + GGML_UNARY_OP_GELU):
            return true;
        default:
            return false;
    }
}

static_assert(isQnnSupportedOp(GGML_OP_NONE), "GGML_OP_NONE is not true");
static_assert(isQnnSupportedOp(GGML_OP_ADD), "GGML_OP_ADD is not true");
static_assert(isQnnSupportedOp(GGML_OP_MUL), "GGML_OP_MUL is not true");
static_assert(isQnnSupportedOp(GGML_OP_MUL_MAT),
              "GGML_OP_MUL_MAT is not true, please check the kQnnSupportedOps table in the backend-ops.cpp file");
static_assert(isQnnSupportedOp(GGML_OP_RESHAPE), "GGML_OP_RESHAPE is not true");
static_assert(!isQnnSupportedOp(GGML_OP_VIEW), "GGML_OP_VIEW is not false");

bool ggml_qnn_supports_tensor(ggml_backend_qnn_device_context *ctx, const ggml_tensor *tensor) {
    if (!tensor) {
        QNN_LOG_DEBUG("tensor is nullptr");
        return false;
    }

#ifndef NDEBUG
    if (tensor->view_src) {
        auto *src_tensor = tensor->view_src;
        QNN_LOG_DEBUG("[%s]tensor(%s_%dx%dx%dx%d) is a view, src: %s_%dx%dx%dx%d", qnn::get_backend_name(ctx->device),
                      ggml_get_name(tensor), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                      ggml_get_name(src_tensor), src_tensor->ne[0], src_tensor->ne[1], src_tensor->ne[2],
                      src_tensor->ne[3]);
    }
#endif

    switch (tensor->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            if (!(ctx->supported_types & (1 << tensor->type))) {
                QNN_LOG_DEBUG("[%s]unsupported data type %s, supported_types: 0x%x", qnn::get_backend_name(ctx->device),
                              ggml_type_name(tensor->type), ctx->supported_types);
                return false;
            }
            break;
        default:
            QNN_LOG_DEBUG("[%s]unsupported data type %s", qnn::get_backend_name(ctx->device),
                          ggml_type_name(tensor->type));
            return false;
    }

    return true;
}

bool ggnl_qnn_supports_op_tensor(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op) {
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!ggml_qnn_supports_tensor(ctx, op)) {
        return false;
    }

    const auto param_count = qnn::get_qnn_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        if (!ggml_qnn_supports_tensor(ctx, op->src[i])) {
            return false;
        }
    }

    return true;
}

bool ggml_qnn_supports_matmul_op(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op) {
    constexpr const size_t kMaxNpuTensorSize = 8192L * 2048 + 8192 * 512 + 2048 * 512; // TODO: fix this
    constexpr const auto get_tensor_size = [](const ggml_tensor *tensor) -> size_t {
        return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    };

    auto *src0 = op->src[0];
    auto *src1 = op->src[1];
    switch (ctx->device) {
        case QNN_BACKEND_NPU:
            if (src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3]) {
                /*
                 * TODO: remove the blocker here when NPU backend supports mul_mat like this:
                 *   [ne03, ne02, n, k] * [ne03 * x, ne02 * y, m, k] -> [ne03 * x, ne02 * y, m, n]
                 */
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]src0 and src1 dimensions are not equal");
                return false;
            } else if (get_tensor_size(src0) + get_tensor_size(src1) + get_tensor_size(op) >= kMaxNpuTensorSize) {
                // TODO: fix this
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]tensor size is too large");
                return false;
            }
            // fall through, from test here, the convert op is super slow on NPU:
            //   https://github.com/usefulsensors/qc_npu_benchmark
        case QNN_BACKEND_GPU:
            if (src0->type != src1->type || src0->type != op->type) {
                // there's no convert op for GPU.
                QNN_LOG_DEBUG("[qnn-gpu][MUL_MAT]type src0(%s), src1(%s) and op(%s) are not equal",
                              qnn::get_ggml_type_name(src0->type), qnn::get_ggml_type_name(src1->type),
                              qnn::get_ggml_type_name(op->type));
                return false;
            }
            break;
        default:
            break;
    }

    if ((src1->ne[2] % src0->ne[2]) != 0 || (src1->ne[3] % src0->ne[3]) != 0) {
        QNN_LOG_DEBUG("[%s][MUL_MAT]src0 and src1 dimensions are not equal", qnn::get_backend_name(ctx->device));
        return false;
    }

    QNN_LOG_DEBUG("[%s][MUL_MAT]supported matmul op", qnn::get_backend_name(ctx->device));
    return true;
}

} // namespace

namespace qnn {

bool device_supports_op(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op) {
    // Note that this function could be called before the device context is initialized
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!isQnnSupportedOp(qnn::get_qnn_op_index(op))) {
#ifndef NDEBUG
        std::string op_key;
        get_graph_key_from_op(op, op_key);
        QNN_LOG_DEBUG("[%s][%s]unsupported op, support/unsupported: %d/%d", qnn::get_backend_name(ctx->device),
                      op_key.c_str(), int(ctx->support_op_count.load()), int(++(ctx->unsupported_op_count)));
#endif
        return false;
    }

    if (!ggnl_qnn_supports_op_tensor(ctx, op)) {
#ifndef NDEBUG
        std::string tensor_dims;
        append_tensor_dimensions(op, tensor_dims);
        QNN_LOG_DEBUG("[%s][%s]unsupported tensor(%s), support/unsupported: %d/%d", qnn::get_backend_name(ctx->device),
                      ggml_op_name(op->op), tensor_dims.c_str(), int(ctx->support_op_count.load()),
                      int(++(ctx->unsupported_op_count)));
#endif
        return false;
    }

    if (op->op == GGML_OP_UNARY) {
        const auto unary_op = ggml_get_unary_op(op);
        if (unary_op == GGML_UNARY_OP_GELU) {
            // TODO: fix this
            QNN_LOG_DEBUG("[%s][GELU]unsupported unary op GGML_UNARY_OP_GELU for NPU, support/unsupported: %d/%d",
                          qnn::get_backend_name(ctx->device), int(ctx->support_op_count.load()),
                          int(++(ctx->unsupported_op_count)));
            return false;
        }
    } else {
        auto *src0 = op->src[0];
        auto *src1 = op->src[1];
        switch (op->op) {
            case GGML_OP_ADD:
                if (!ggml_are_same_shape(src0, src1)) {
                    QNN_LOG_DEBUG("[%s][ADD] src0 and src1 dimensions are not equal, support/unsupported: %d/%d",
                                  qnn::get_backend_name(ctx->device), int(ctx->support_op_count.load()),
                                  int(++(ctx->unsupported_op_count)));
                    return false;
                }
                break;

            case GGML_OP_MUL_MAT:
                if (ggml_qnn_supports_matmul_op(ctx, op)) {
                    QNN_LOG_DEBUG("[%s][MUL_MAT]supported matmul op, support/unsupported: %d/%d",
                                  qnn::get_backend_name(ctx->device), int(++(ctx->support_op_count)),
                                  int(ctx->unsupported_op_count.load()));
                    return true;
                } else {
                    QNN_LOG_DEBUG("[%s][MUL_MAT]unsupported matmul op, support/unsupported: %d/%d",
                                  qnn::get_backend_name(ctx->device), int(ctx->support_op_count.load()),
                                  int(++(ctx->unsupported_op_count)));
                    return false;
                }

            default:
                return false;
        }
    }

    return true;
}

bool device_compute_graph(ggml_backend_qnn_device_context *ctx, ggml_cgraph *cgraph) {
    QNN_LOG_DEBUG("[%s]compute graph start, nodes count: %d", qnn::get_backend_name(ctx->device), (int)cgraph->n_nodes);

    auto qnn_graph = get_qnn_graph_from_cache(ctx, cgraph);
    bool success = qnn_graph && qnn_graph->execute(cgraph);

    QNN_LOG_DEBUG("[%s]compute graph, success: %d", qnn::get_backend_name(ctx->device), (int)success);
    return success;
}

} // namespace qnn
