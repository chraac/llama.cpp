
#include "backend-ops.hpp"

#include <memory>

#include "ggml-impl.h"

#include "graph.hpp"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

bool qnn_is_op_valid(ggml_backend_qnn_device_context *ctx, const ggml_tensor *dst) {
    if (!ctx || !dst) {
        QNN_LOG_WARN("invalid params");
        return false;
    }

    auto instance = ctx->instance;
    if (!instance) {
        QNN_LOG_WARN("invalid instance");
        return false;
    }

    const auto param_count = qnn::get_qnn_op_input_param_count(dst);
    switch (param_count) {
        case 1:
            return dst->src[0];
        case 2:
            return dst->src[0] && dst->src[1];
        default:
            QNN_LOG_WARN("invalid op param count %d", (int)param_count);
            break;
    }

    return false;
}

#ifndef NDEBUG
void print_ggml_tensor(const ggml_tensor *tensor) {
    QNN_LOG_DEBUG("%s: type:%s ne: %ldx%ldx%ldx%ld, nb: %ldx%ldx%ldx%ld", tensor->name, ggml_type_name(tensor->type),
                  (long)tensor->ne[0], (long)tensor->ne[1], (long)tensor->ne[2], (long)tensor->ne[3],
                  (long)tensor->nb[0], (long)tensor->nb[1], (long)tensor->nb[2], (long)tensor->nb[3]);
}
#endif

} // namespace

namespace {

typedef bool (*ggml_qnn_op_t)(ggml_backend_qnn_device_context *ctx, ggml_tensor *dst);

bool execute_graph(qnn::qnn_graph *graph, ggml_tensor *output) {
    if (!graph->execute(output)) {
        QNN_LOG_WARN("execute failed");
        return false;
    }

    return true;
}

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

qnn::qnn_graph *get_qnn_graph_from_cache(ggml_backend_qnn_device_context *ctx, ggml_tensor *output) {
    auto &graph_cache = ctx->qnn_graph_cache;
    std::string graph_key;
    get_graph_key_from_op(output, graph_key);
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

        if (!graph->build_graph_from_op(output)) {
            QNN_LOG_ERROR("[%s]build_graph_from_op failed", qnn::get_backend_name(ctx->device));
            return nullptr;
        }

        graph_ptr = graph.get();
        graph_cache[graph_key] = std::move(graph);
    }

    return graph_ptr;
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
            QNN_LOG_ERROR("[%s]build_graph_from_op failed", qnn::get_backend_name(ctx->device));
            return nullptr;
        }

        graph_ptr = graph.get();
        graph_cache[graph_key] = std::move(graph);
    }

    return graph_ptr;
}

bool qnn_generic_op_impl(ggml_backend_qnn_device_context *ctx, ggml_tensor *dst) {
    if (!qnn_is_op_valid(ctx, dst)) {
        return false;
    }

    auto *graph_ptr = get_qnn_graph_from_cache(ctx, dst);
    bool succeed = graph_ptr && execute_graph(graph_ptr, dst);

#ifndef NDEBUG
    if (!succeed) {
        const auto param_count = qnn::get_qnn_op_input_param_count(dst);
        for (size_t i = 0; i < param_count; ++i) {
            print_ggml_tensor(dst->src[i]);
        }
        print_ggml_tensor(dst);
    }
#endif

    return succeed;
}

bool qnn_nop_impl(ggml_backend_qnn_device_context *ctx, ggml_tensor *dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return true;
}

constexpr const ggml_qnn_op_t kQnnOpsTable[] = {
    qnn_nop_impl,        // GGML_OP_NONE
    nullptr,             // GGML_OP_DUP
    qnn_generic_op_impl, // GGML_OP_ADD
    nullptr,             // GGML_OP_ADD1
    nullptr,             // GGML_OP_ACC
    qnn_generic_op_impl, // GGML_OP_SUB
    qnn_generic_op_impl, // GGML_OP_MUL
    qnn_generic_op_impl, // GGML_OP_DIV
    nullptr,             // GGML_OP_SQR
    qnn_generic_op_impl, // GGML_OP_SQRT
    qnn_generic_op_impl, // GGML_OP_LOG
    nullptr,             // GGML_OP_SIN
    nullptr,             // GGML_OP_COS
    nullptr,             // GGML_OP_SUM
    nullptr,             // GGML_OP_SUM_ROWS
    nullptr,             // GGML_OP_MEAN
    nullptr,             // GGML_OP_ARGMAX
    nullptr,             // GGML_OP_COUNT_EQUAL
    nullptr,             // GGML_OP_REPEAT
    nullptr,             // GGML_OP_REPEAT_BACK
    nullptr,             // GGML_OP_CONCAT
    nullptr,             // GGML_OP_SILU_BACK
    nullptr,             // GGML_OP_NORM
    nullptr,             // GGML_OP_RMS_NORM
    nullptr,             // GGML_OP_RMS_NORM_BACK
    nullptr,             // GGML_OP_GROUP_NORM

    qnn_generic_op_impl, // GGML_OP_MUL_MAT
    nullptr,             // GGML_OP_MUL_MAT_ID
    nullptr,             // GGML_OP_OUT_PROD

    nullptr,      // GGML_OP_SCALE
    nullptr,      // GGML_OP_SET
    nullptr,      // GGML_OP_CPY
    nullptr,      // GGML_OP_CONT
    qnn_nop_impl, // GGML_OP_RESHAPE
    nullptr,      // GGML_OP_VIEW
    nullptr,      // GGML_OP_PERMUTE
    nullptr,      // GGML_OP_TRANSPOSE
    nullptr,      // GGML_OP_GET_ROWS
    nullptr,      // GGML_OP_GET_ROWS_BACK
    nullptr,      // GGML_OP_DIAG
    nullptr,      // GGML_OP_DIAG_MASK_INF
    nullptr,      // GGML_OP_DIAG_MASK_ZERO
    nullptr,      // GGML_OP_SOFT_MAX
    nullptr,      // GGML_OP_SOFT_MAX_BACK
    nullptr,      // GGML_OP_ROPE
    nullptr,      // GGML_OP_ROPE_BACK
    nullptr,      // GGML_OP_CLAMP
    nullptr,      // GGML_OP_CONV_TRANSPOSE_1D
    nullptr,      // GGML_OP_IM2COL
    nullptr,      // GGML_OP_IM2COL_BACK
    nullptr,      // GGML_OP_CONV_TRANSPOSE_2D
    nullptr,      // GGML_OP_POOL_1D
    nullptr,      // GGML_OP_POOL_2D
    nullptr,      // GGML_OP_POOL_2D_BACK
    nullptr,      // GGML_OP_UPSCALE
    nullptr,      // GGML_OP_PAD
    nullptr,      // GGML_OP_PAD_REFLECT_1D
    nullptr,      // GGML_OP_ARANGE
    nullptr,      // GGML_OP_TIMESTEP_EMBEDDING
    nullptr,      // GGML_OP_ARGSORT
    nullptr,      // GGML_OP_LEAKY_RELU

    nullptr, // GGML_OP_FLASH_ATTN_EXT
    nullptr, // GGML_OP_FLASH_ATTN_BACK
    nullptr, // GGML_OP_SSM_CONV
    nullptr, // GGML_OP_SSM_SCAN
    nullptr, // GGML_OP_WIN_PART
    nullptr, // GGML_OP_WIN_UNPART
    nullptr, // GGML_OP_GET_REL_POS
    nullptr, // GGML_OP_ADD_REL_POS
    nullptr, // GGML_OP_RWKV_WKV6
    nullptr, // GGML_OP_GATED_LINEAR_ATTN

    nullptr, // GGML_OP_UNARY

    nullptr, // GGML_OP_MAP_UNARY
    nullptr, // GGML_OP_MAP_BINARY

    nullptr, // GGML_OP_MAP_CUSTOM1_F32
    nullptr, // GGML_OP_MAP_CUSTOM2_F32
    nullptr, // GGML_OP_MAP_CUSTOM3_F32

    nullptr, // GGML_OP_MAP_CUSTOM1
    nullptr, // GGML_OP_MAP_CUSTOM2
    nullptr, // GGML_OP_MAP_CUSTOM3

    nullptr, // GGML_OP_CROSS_ENTROPY_LOSS
    nullptr, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    nullptr, // GGML_OP_OPT_STEP_ADAMW

    // ggml_unary_op
    nullptr,             // GGML_UNARY_OP_ABS
    nullptr,             // GGML_UNARY_OP_SGN
    nullptr,             // GGML_UNARY_OP_NEG
    nullptr,             // GGML_UNARY_OP_STEP
    nullptr,             // GGML_UNARY_OP_TANH
    nullptr,             // GGML_UNARY_OP_ELU
    nullptr,             // GGML_UNARY_OP_RELU
    nullptr,             // GGML_UNARY_OP_SIGMOID
    qnn_generic_op_impl, // GGML_UNARY_OP_GELU
    nullptr,             // GGML_UNARY_OP_GELU_QUICK
    nullptr,             // GGML_UNARY_OP_SILU
    nullptr,             // GGML_UNARY_OP_HARDSWISH
    nullptr,             // GGML_UNARY_OP_HARDSIGMOID
    nullptr,             // GGML_UNARY_OP_EXP
};

static_assert(kQnnOpsTable[GGML_OP_NONE] == qnn_nop_impl, "GGML_OP_NONE does not match the qnn_nop_impl function");
static_assert(kQnnOpsTable[GGML_OP_ADD] == qnn_generic_op_impl,
              "GGML_OP_ADD does not match the qnn_generic_op_impl function");
static_assert(kQnnOpsTable[GGML_OP_MUL] == qnn_generic_op_impl,
              "GGML_OP_MUL does not match the qnn_generic_op_impl function");
static_assert(kQnnOpsTable[GGML_OP_MUL_MAT] == qnn_generic_op_impl,
              "GGML_OP_MUL_MAT does not match the qnn_generic_op_impl function");
static_assert(kQnnOpsTable[GGML_OP_RESHAPE] == qnn_nop_impl,
              "GGML_OP_RESHAPE does not match the qnn_nop_impl function");
static_assert(kQnnOpsTable[GGML_OP_VIEW] == nullptr, "GGML_OP_VIEW is not nullptr");
static_assert(std::size(kQnnOpsTable) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kQnnOpsTable table");

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
            if (!(ctx->supported_types & (uint64_t(1) << tensor->type))) {
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
    constexpr const size_t kMaxNpuTensorSize = 8192L * 2048 + 8192 * 512 + 2048 * 512;
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
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]src0 and src1 dimensions are not equal, support/unsupported: %d/%d",
                              ctx->support_op_count.load(), ++(ctx->unsupported_op_count));
                return false;
            } else if (get_tensor_size(src0) + get_tensor_size(src1) + get_tensor_size(op) >= kMaxNpuTensorSize) {
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]tensor size is too large, support/unsupported: %d/%d",
                              ctx->support_op_count.load(), ++(ctx->unsupported_op_count));
                return false;
            }
            // fall through, from test here, the convert op is super slow on NPU:
            //   https://github.com/usefulsensors/qc_npu_benchmark
        case QNN_BACKEND_GPU:
            if (src0->type != src1->type || src0->type != op->type) {
                // there's no convert op for GPU.
                QNN_LOG_DEBUG(
                    "[qnn-gpu][MUL_MAT]type src0(%d), src1(%d) and op(%d) are not equal, support/unsupported: %d/%d",
                    src0->type, src1->type, op->type, ctx->support_op_count.load(), ++(ctx->unsupported_op_count));
                return false;
            }
            break;
        default:
            break;
    }

    if ((src1->ne[2] % src0->ne[2]) != 0 || (src1->ne[3] % src0->ne[3]) != 0) {
        QNN_LOG_DEBUG("[%s][MUL_MAT]src0 and src1 dimensions are not equal, support/unsupported: %d/%d",
                      qnn::get_backend_name(ctx->device), ctx->support_op_count.load(), ++(ctx->unsupported_op_count));
        return false;
    }

    QNN_LOG_DEBUG("[%s][MUL_MAT]supported matmul op, support/unsupported: %d/%d", qnn::get_backend_name(ctx->device),
                  ++(ctx->support_op_count), ctx->unsupported_op_count.load());
    return true;
}

} // namespace

namespace qnn {

bool device_supports_op(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op) {
    // Note that this function could be called before the device context is initialized
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!kQnnOpsTable[qnn::get_qnn_op_index(op)]) {
#ifndef NDEBUG
        std::string op_key;
        get_graph_key_from_op(op, op_key);
        QNN_LOG_DEBUG("[%s]unsupported op", op_key.c_str());
#endif
        return false;
    }

    if (!ggnl_qnn_supports_op_tensor(ctx, op)) {
#ifndef NDEBUG
        std::string tensor_dims;
        append_tensor_dimensions(op, tensor_dims);
        QNN_LOG_DEBUG("[%s]unsupported tensor(%s)", ggml_op_name(op->op), tensor_dims.c_str());
#endif
        return false;
    }

    if (op->op == GGML_OP_UNARY) {
        const auto unary_op = ggml_get_unary_op(op);
        if (unary_op == GGML_UNARY_OP_GELU) {
            // TODO: fix this
            QNN_LOG_DEBUG("[GELU]unsupported unary op GGML_UNARY_OP_GELU for NPU");
            return false;
        }
    } else {
        auto *src0 = op->src[0];
        auto *src1 = op->src[1];
        switch (op->op) {
            case GGML_OP_ADD:
                if (!ggml_are_same_shape(src0, src1)) {
                    QNN_LOG_DEBUG("[ADD] src0 and src1 dimensions are not equal");
                    return false;
                }
                break;

            case GGML_OP_MUL_MAT:
                return ggml_qnn_supports_matmul_op(ctx, op);

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
