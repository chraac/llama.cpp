
#include "backend-ops.hpp"

#include <memory>

#include "graph.hpp"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#ifndef NDEBUG

namespace {

bool qnn_is_valid_params(ggml_backend_qnn_device_context *ctx, const ggml_tensor *src, ggml_tensor *dst) {
    if (!ctx || !src || !dst) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    auto instance = ctx->instance;
    if (!instance) {
        QNN_LOG_WARN("invalid instance\n");
        return false;
    }

    return true;
}

bool qnn_is_valid_params(ggml_backend_qnn_device_context *ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                         ggml_tensor *dst) {
    if (!ctx || !src0 || !src1 || !dst) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    auto instance = ctx->instance;
    if (!instance) {
        QNN_LOG_WARN("invalid instance\n");
        return false;
    }

    return true;
}

void print_ggml_tensor(const ggml_tensor *tensor) {
    QNN_LOG_DEBUG("%s: type:%s ne: %ldx%ldx%ldx%ld, nb: %ldx%ldx%ldx%ld\n", tensor->name, ggml_type_name(tensor->type),
                  (long)tensor->ne[0], (long)tensor->ne[1], (long)tensor->ne[2], (long)tensor->ne[3],
                  (long)tensor->nb[0], (long)tensor->nb[1], (long)tensor->nb[2], (long)tensor->nb[3]);
}

} // namespace

#define CHECK_PARAMS(ctx, ...)                      \
    if (!qnn_is_valid_params((ctx), __VA_ARGS__)) { \
        return false;                               \
    }

#else
#define CHECK_PARAMS(ctx, ...)
#endif

namespace {

bool is_tensor_dimensions_equal(const ggml_tensor *l, const ggml_tensor *r) {
    const auto dim_l = ggml_n_dims(l);
    if (dim_l != ggml_n_dims(r)) {
        return false;
    }

    for (int i = 0; i < dim_l; i++) {
        if (l->ne[i] != r->ne[i]) {
            return false;
        }
    }

    return true;
}

typedef bool (*ggml_qnn_unary_op_t)(ggml_backend_qnn_device_context *ctx, ggml_tensor *src, ggml_tensor *dst);
typedef bool (*ggml_qnn_binary_op_t)(ggml_backend_qnn_device_context *ctx, ggml_tensor *src0, ggml_tensor *src1,
                                     ggml_tensor *dst);

typedef const ggml_qnn_unary_op_t (&ggml_qnn_unary_op_array_t)[GGML_OP_COUNT + GGML_UNARY_OP_COUNT];
typedef const ggml_qnn_binary_op_t (&ggml_qnn_binary_op_array_t)[GGML_OP_COUNT];

constexpr const size_t kGgmlUnaryOpStart = GGML_OP_COUNT;

template <size_t _Size>
qnn::ggml_tensor_array_t to_ggml_tensor_array(const std::array<ggml_tensor *, _Size> &array) {
    return qnn::ggml_tensor_array_t(array.data(), array.data() + _Size);
}

template <size_t _InputSize>
bool execute_graph(qnn::ggml_qnn_graph *graph, const std::array<ggml_tensor *, _InputSize> &inputs,
                   ggml_tensor *output) {
    if (!graph->execute(to_ggml_tensor_array<_InputSize>(inputs), to_ggml_tensor_array<1>({output}))) {
        QNN_LOG_WARN("execute failed\n");
        return false;
    }

    return true;
}

template <size_t _InputSize, size_t _OutputSize>
std::string get_graph_key(const std::string &op_name, const std::array<ggml_tensor *, _InputSize> &inputs,
                          const std::array<ggml_tensor *, _OutputSize> &outputs) {
    constexpr static const auto append_dimensions = [](std::string &key, const ggml_tensor *tensor) {
        char buffer[256] = {};
        snprintf(buffer, sizeof(buffer), "_%ldx%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                 (long)tensor->ne[2], (long)tensor->ne[3], qnn::get_ggml_type_name(tensor->type));
        key += buffer;
    };

    std::string graph_key(op_name);
    for (auto &input : inputs) {
        append_dimensions(graph_key, input);
    }

    graph_key += qnn::get_ggml_type_name(outputs.front()->type);
    return graph_key;
}

constexpr const char *kGgmlOpToQnnOp[] = {
    nullptr,                         // GGML_OP_NONE
    nullptr,                         // GGML_OP_DUP
    QNN_OP_ELEMENT_WISE_ADD,         // GGML_OP_ADD
    nullptr,                         // GGML_OP_ADD1
    nullptr,                         // GGML_OP_ACC
    QNN_OP_ELEMENT_WISE_SUBTRACT,    // GGML_OP_SUB
    QNN_OP_ELEMENT_WISE_MULTIPLY,    // GGML_OP_MUL
    QNN_OP_ELEMENT_WISE_DIVIDE,      // GGML_OP_DIV
    nullptr,                         // GGML_OP_SQR
    QNN_OP_ELEMENT_WISE_SQUARE_ROOT, // GGML_OP_SQRT
    QNN_OP_ELEMENT_WISE_LOG,         // GGML_OP_LOG
    nullptr,                         // GGML_OP_SIN
    nullptr,                         // GGML_OP_COS
    nullptr,                         // GGML_OP_SUM
    nullptr,                         // GGML_OP_SUM_ROWS
    nullptr,                         // GGML_OP_MEAN
    nullptr,                         // GGML_OP_ARGMAX
    nullptr,                         // GGML_OP_COUNT_EQUAL
    nullptr,                         // GGML_OP_REPEAT
    nullptr,                         // GGML_OP_REPEAT_BACK
    nullptr,                         // GGML_OP_CONCAT
    nullptr,                         // GGML_OP_SILU_BACK
    nullptr,                         // GGML_OP_NORM
    nullptr,                         // GGML_OP_RMS_NORM
    nullptr,                         // GGML_OP_RMS_NORM_BACK
    nullptr,                         // GGML_OP_GROUP_NORM

    QNN_OP_MAT_MUL, // GGML_OP_MUL_MAT
    nullptr,        // GGML_OP_MUL_MAT_ID
    nullptr,        // GGML_OP_OUT_PROD

    nullptr,          // GGML_OP_SCALE
    nullptr,          // GGML_OP_SET
    nullptr,          // GGML_OP_CPY
    nullptr,          // GGML_OP_CONT
    nullptr,          // GGML_OP_RESHAPE
    nullptr,          // GGML_OP_VIEW
    QNN_OP_TRANSPOSE, // GGML_OP_PERMUTE
    nullptr,          // GGML_OP_TRANSPOSE
    nullptr,          // GGML_OP_GET_ROWS
    nullptr,          // GGML_OP_GET_ROWS_BACK
    nullptr,          // GGML_OP_DIAG
    nullptr,          // GGML_OP_DIAG_MASK_INF
    nullptr,          // GGML_OP_DIAG_MASK_ZERO
    nullptr,          // GGML_OP_SOFT_MAX
    nullptr,          // GGML_OP_SOFT_MAX_BACK
    nullptr,          // GGML_OP_ROPE
    nullptr,          // GGML_OP_ROPE_BACK
    nullptr,          // GGML_OP_CLAMP
    nullptr,          // GGML_OP_CONV_TRANSPOSE_1D
    nullptr,          // GGML_OP_IM2COL
    nullptr,          // GGML_OP_IM2COL_BACK
    nullptr,          // GGML_OP_CONV_TRANSPOSE_2D
    nullptr,          // GGML_OP_POOL_1D
    nullptr,          // GGML_OP_POOL_2D
    nullptr,          // GGML_OP_POOL_2D_BACK
    nullptr,          // GGML_OP_UPSCALE
    nullptr,          // GGML_OP_PAD
    nullptr,          // GGML_OP_ARANGE
    nullptr,          // GGML_OP_TIMESTEP_EMBEDDING
    nullptr,          // GGML_OP_ARGSORT
    nullptr,          // GGML_OP_LEAKY_RELU

    nullptr, // GGML_OP_FLASH_ATTN_EXT
    nullptr, // GGML_OP_FLASH_ATTN_BACK
    nullptr, // GGML_OP_SSM_CONV
    nullptr, // GGML_OP_SSM_SCAN
    nullptr, // GGML_OP_WIN_PART
    nullptr, // GGML_OP_WIN_UNPART
    nullptr, // GGML_OP_GET_REL_POS
    nullptr, // GGML_OP_ADD_REL_POS
    nullptr, // GGML_OP_RWKV_WKV

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
    nullptr,     // GGML_UNARY_OP_ABS
    nullptr,     // GGML_UNARY_OP_SGN
    nullptr,     // GGML_UNARY_OP_NEG
    nullptr,     // GGML_UNARY_OP_STEP
    nullptr,     // GGML_UNARY_OP_TANH
    nullptr,     // GGML_UNARY_OP_ELU
    nullptr,     // GGML_UNARY_OP_RELU
    nullptr,     // GGML_UNARY_OP_SIGMOID
    QNN_OP_GELU, // GGML_UNARY_OP_GELU
    nullptr,     // GGML_UNARY_OP_GELU_QUICK
    nullptr,     // GGML_UNARY_OP_SILU
    nullptr,     // GGML_UNARY_OP_HARDSWISH
    nullptr,     // GGML_UNARY_OP_HARDSIGMOID
    nullptr,     // GGML_UNARY_OP_EXP
};

static_assert(sizeof(kGgmlOpToQnnOp) / sizeof(kGgmlOpToQnnOp[0]) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kGgmlOpToQnnOp table");
static_assert(kGgmlOpToQnnOp[GGML_UNARY_OP_GELU + kGgmlUnaryOpStart] != nullptr,
              "GGML_UNARY_OP_GELU does not correspond to QNN_OP_GELU");

template <size_t _InputSize>
qnn::ggml_qnn_graph *get_qnn_graph_from_cache(ggml_backend_qnn_device_context *ctx, size_t op,
                                              const std::array<ggml_tensor *, _InputSize> &inputs,
                                              ggml_tensor *output) {
    GGML_ASSERT(op < (GGML_OP_COUNT + GGML_UNARY_OP_COUNT));

    auto &graph_cache = ctx->qnn_graph_cache;
    const auto *op_name =
        op < kGgmlUnaryOpStart ? ggml_op_name(ggml_op(op)) : ggml_unary_op_name(ggml_unary_op(op - kGgmlUnaryOpStart));
    auto graph_key = get_graph_key<_InputSize, 1>(op_name, inputs, {output});
    auto it = graph_cache.find(graph_key);
    qnn::ggml_qnn_graph *graph_ptr = nullptr;
    if (it != graph_cache.end()) {
        QNN_LOG_DEBUG("found graph %s in cache\n", graph_key.c_str());
        graph_ptr = it->second.get();
    } else {
        auto graph =
            std::make_unique<qnn::ggml_qnn_graph>(graph_key, ctx->device, ctx->instance, ctx->socinfo.vtcm_size_in_mb);
        if (!graph->is_valid()) {
            return nullptr;
        }

        auto op_constructor = qnn::create_op_constructor(kGgmlOpToQnnOp[op]);
        if (!graph->build_graph(op_constructor, to_ggml_tensor_array<_InputSize>(inputs),
                                to_ggml_tensor_array<1>({output}))) {
            QNN_LOG_ERROR("build_graph failed\n");
            return nullptr;
        }

        graph_ptr = graph.get();
        graph_cache[graph_key] = std::move(graph);
    }

    return graph_ptr;
}

template <ggml_op _GgmlOp>
bool qnn_binary_op_impl(ggml_backend_qnn_device_context *ctx, ggml_tensor *src0, ggml_tensor *src1, ggml_tensor *dst) {
    static_assert(kGgmlOpToQnnOp[_GgmlOp] != nullptr, "GGML_OP does not have a corresponding QNN_OP");

    CHECK_PARAMS(ctx, src0, src1, dst);

    bool succeed = false;
    auto *graph_ptr = get_qnn_graph_from_cache<2>(ctx, _GgmlOp, {src0, src1}, dst);
    if (graph_ptr) {
        succeed = execute_graph<2>(graph_ptr, {src0, src1}, dst);
    }

#ifndef NDEBUG
    if (!succeed) {
        print_ggml_tensor(src0);
        print_ggml_tensor(src1);
        print_ggml_tensor(dst);
    }
#endif

    return succeed;
}

template <size_t _GgmlOp>
bool qnn_unary_op_impl(ggml_backend_qnn_device_context *ctx, ggml_tensor *src, ggml_tensor *dst) {
    static_assert(kGgmlOpToQnnOp[_GgmlOp] != nullptr, "GGML_OP does not have a corresponding QNN_OP");

    CHECK_PARAMS(ctx, src, dst);

    bool succeed = false;
    auto *graph_ptr = get_qnn_graph_from_cache<1>(ctx, _GgmlOp, {src}, dst);
    if (graph_ptr) {
        succeed = execute_graph<1>(graph_ptr, {src}, dst);
    }

#ifndef NDEBUG
    if (!succeed) {
        print_ggml_tensor(src);
        print_ggml_tensor(dst);
    }
#endif

    return succeed;
}

bool qnn_unary_nop_impl(ggml_backend_qnn_device_context *ctx, ggml_tensor *src, ggml_tensor *dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return true;
}

bool qnn_binary_nop_impl(ggml_backend_qnn_device_context *ctx, ggml_tensor *src0, ggml_tensor *src1, ggml_tensor *dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return true;
}

constexpr const ggml_qnn_unary_op_t kQnnUnaryOpsTable[] = {
    nullptr,                         // GGML_OP_NONE
    nullptr,                         // GGML_OP_DUP
    nullptr,                         // GGML_OP_ADD
    nullptr,                         // GGML_OP_ADD1
    nullptr,                         // GGML_OP_ACC
    nullptr,                         // GGML_OP_SUB
    nullptr,                         // GGML_OP_MUL
    nullptr,                         // GGML_OP_DIV
    nullptr,                         // GGML_OP_SQR
    qnn_unary_op_impl<GGML_OP_SQRT>, // GGML_OP_SQRT
    qnn_unary_op_impl<GGML_OP_LOG>,  // GGML_OP_LOG
    nullptr,                         // GGML_OP_SIN
    nullptr,                         // GGML_OP_COS
    nullptr,                         // GGML_OP_SUM
    nullptr,                         // GGML_OP_SUM_ROWS
    nullptr,                         // GGML_OP_MEAN
    nullptr,                         // GGML_OP_ARGMAX
    nullptr,                         // GGML_OP_COUNT_EQUAL
    nullptr,                         // GGML_OP_REPEAT
    nullptr,                         // GGML_OP_REPEAT_BACK
    nullptr,                         // GGML_OP_CONCAT
    nullptr,                         // GGML_OP_SILU_BACK
    nullptr,                         // GGML_OP_NORM
    nullptr,                         // GGML_OP_RMS_NORM
    nullptr,                         // GGML_OP_RMS_NORM_BACK
    nullptr,                         // GGML_OP_GROUP_NORM

    nullptr, // GGML_OP_MUL_MAT
    nullptr, // GGML_OP_MUL_MAT_ID
    nullptr, // GGML_OP_OUT_PROD

    nullptr,                            // GGML_OP_SCALE
    nullptr,                            // GGML_OP_SET
    nullptr,                            // GGML_OP_CPY
    nullptr,                            // GGML_OP_CONT
    nullptr,                            // GGML_OP_RESHAPE
    qnn_unary_nop_impl,                 // GGML_OP_VIEW
    qnn_unary_op_impl<GGML_OP_PERMUTE>, // GGML_OP_PERMUTE
    nullptr,                            // GGML_OP_TRANSPOSE
    qnn_unary_nop_impl,                 // GGML_OP_GET_ROWS
    nullptr,                            // GGML_OP_GET_ROWS_BACK
    nullptr,                            // GGML_OP_DIAG
    nullptr,                            // GGML_OP_DIAG_MASK_INF
    nullptr,                            // GGML_OP_DIAG_MASK_ZERO
    nullptr,                            // GGML_OP_SOFT_MAX
    nullptr,                            // GGML_OP_SOFT_MAX_BACK
    nullptr,                            // GGML_OP_ROPE
    nullptr,                            // GGML_OP_ROPE_BACK
    nullptr,                            // GGML_OP_CLAMP
    nullptr,                            // GGML_OP_CONV_TRANSPOSE_1D
    nullptr,                            // GGML_OP_IM2COL
    nullptr,                            // GGML_OP_IM2COL_BACK
    nullptr,                            // GGML_OP_CONV_TRANSPOSE_2D
    nullptr,                            // GGML_OP_POOL_1D
    nullptr,                            // GGML_OP_POOL_2D
    nullptr,                            // GGML_OP_POOL_2D_BACK
    nullptr,                            // GGML_OP_UPSCALE
    nullptr,                            // GGML_OP_PAD
    nullptr,                            // GGML_OP_ARANGE
    nullptr,                            // GGML_OP_TIMESTEP_EMBEDDING
    nullptr,                            // GGML_OP_ARGSORT
    nullptr,                            // GGML_OP_LEAKY_RELU

    nullptr, // GGML_OP_FLASH_ATTN_EXT
    nullptr, // GGML_OP_FLASH_ATTN_BACK
    nullptr, // GGML_OP_SSM_CONV
    nullptr, // GGML_OP_SSM_SCAN
    nullptr, // GGML_OP_WIN_PART
    nullptr, // GGML_OP_WIN_UNPART
    nullptr, // GGML_OP_GET_REL_POS
    nullptr, // GGML_OP_ADD_REL_POS
    nullptr, // GGML_OP_RWKV_WKV

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
    nullptr,                                                   // GGML_UNARY_OP_ABS
    nullptr,                                                   // GGML_UNARY_OP_SGN
    nullptr,                                                   // GGML_UNARY_OP_NEG
    nullptr,                                                   // GGML_UNARY_OP_STEP
    nullptr,                                                   // GGML_UNARY_OP_TANH
    nullptr,                                                   // GGML_UNARY_OP_ELU
    nullptr,                                                   // GGML_UNARY_OP_RELU
    nullptr,                                                   // GGML_UNARY_OP_SIGMOID
    qnn_unary_op_impl<GGML_UNARY_OP_GELU + kGgmlUnaryOpStart>, // GGML_UNARY_OP_GELU
    nullptr,                                                   // GGML_UNARY_OP_GELU_QUICK
    nullptr,                                                   // GGML_UNARY_OP_SILU
    nullptr,                                                   // GGML_UNARY_OP_HARDSWISH
    nullptr,                                                   // GGML_UNARY_OP_HARDSIGMOID
    nullptr,                                                   // GGML_UNARY_OP_EXP
};

static_assert(sizeof(kQnnUnaryOpsTable) / sizeof(kQnnUnaryOpsTable[0]) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kQnnUnaryOpsTable table");

static constexpr const ggml_qnn_binary_op_t kQnnBinaryOpsTable[] = {
    nullptr,                         // GGML_OP_NONE
    nullptr,                         // GGML_OP_DUP
    qnn_binary_op_impl<GGML_OP_ADD>, // GGML_OP_ADD
    nullptr,                         // GGML_OP_ADD1
    nullptr,                         // GGML_OP_ACC
    qnn_binary_op_impl<GGML_OP_SUB>, // GGML_OP_SUB
    qnn_binary_op_impl<GGML_OP_MUL>, // GGML_OP_MUL
    qnn_binary_op_impl<GGML_OP_DIV>, // GGML_OP_DIV
    nullptr,                         // GGML_OP_SQR
    nullptr,                         // GGML_OP_SQRT
    nullptr,                         // GGML_OP_LOG
    nullptr,                         // GGML_OP_SIN
    nullptr,                         // GGML_OP_COS
    nullptr,                         // GGML_OP_SUM
    nullptr,                         // GGML_OP_SUM_ROWS
    nullptr,                         // GGML_OP_MEAN
    nullptr,                         // GGML_OP_ARGMAX
    nullptr,                         // GGML_OP_COUNT_EQUAL
    nullptr,                         // GGML_OP_REPEAT
    nullptr,                         // GGML_OP_REPEAT_BACK
    nullptr,                         // GGML_OP_CONCAT
    nullptr,                         // GGML_OP_SILU_BACK
    nullptr,                         // GGML_OP_NORM
    nullptr,                         // GGML_OP_RMS_NORM
    nullptr,                         // GGML_OP_RMS_NORM_BACK
    nullptr,                         // GGML_OP_GROUP_NORM

    qnn_binary_op_impl<GGML_OP_MUL_MAT>, // GGML_OP_MUL_MAT
    nullptr,                             // GGML_OP_MUL_MAT_ID
    nullptr,                             // GGML_OP_OUT_PROD

    nullptr, // GGML_OP_SCALE
    nullptr, // GGML_OP_SET
    nullptr, // GGML_OP_CPY
    nullptr, // GGML_OP_CONT
    nullptr, // GGML_OP_RESHAPE
    nullptr, // GGML_OP_VIEW
    nullptr, // GGML_OP_PERMUTE
    nullptr, // GGML_OP_TRANSPOSE
    nullptr, // GGML_OP_GET_ROWS
    nullptr, // GGML_OP_GET_ROWS_BACK
    nullptr, // GGML_OP_DIAG
    nullptr, // GGML_OP_DIAG_MASK_INF
    nullptr, // GGML_OP_DIAG_MASK_ZERO
    nullptr, // GGML_OP_SOFT_MAX
    nullptr, // GGML_OP_SOFT_MAX_BACK
    nullptr, // GGML_OP_ROPE
    nullptr, // GGML_OP_ROPE_BACK
    nullptr, // GGML_OP_CLAMP
    nullptr, // GGML_OP_CONV_TRANSPOSE_1D
    nullptr, // GGML_OP_IM2COL
    nullptr, // GGML_OP_IM2COL_BACK
    nullptr, // GGML_OP_CONV_TRANSPOSE_2D
    nullptr, // GGML_OP_POOL_1D
    nullptr, // GGML_OP_POOL_2D
    nullptr, // GGML_OP_POOL_2D_BACK
    nullptr, // GGML_OP_UPSCALE
    nullptr, // GGML_OP_PAD
    nullptr, // GGML_OP_ARANGE
    nullptr, // GGML_OP_TIMESTEP_EMBEDDING
    nullptr, // GGML_OP_ARGSORT
    nullptr, // GGML_OP_LEAKY_RELU

    nullptr, // GGML_OP_FLASH_ATTN_EXT
    nullptr, // GGML_OP_FLASH_ATTN_BACK
    nullptr, // GGML_OP_SSM_CONV
    nullptr, // GGML_OP_SSM_SCAN
    nullptr, // GGML_OP_WIN_PART
    nullptr, // GGML_OP_WIN_UNPART
    nullptr, // GGML_OP_GET_REL_POS
    nullptr, // GGML_OP_ADD_REL_POS
    nullptr, // GGML_OP_RWKV_WKV

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
};

static_assert(sizeof(kQnnBinaryOpsTable) / sizeof(kQnnBinaryOpsTable[0]) == GGML_OP_COUNT,
              "GGML_OP_COUNT does not match the size of the kQnnBinaryOpsTable table");

bool ggml_qnn_supports_tensor(ggml_backend_qnn_device_context *ctx, const ggml_tensor *tensor) {
    if (!tensor) {
        QNN_LOG_DEBUG("tensor is nullptr");
        return false;
    }

#ifndef NDEBUG
    auto *type_name = ggml_get_type_traits(tensor->type)->type_name;
#endif
    switch (tensor->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            if (!(ctx->supported_types & (1 << tensor->type))) {
                QNN_LOG_DEBUG("unsupported data type %s for backend %s, supported_types: 0x%x", type_name,
                              qnn::get_backend_name(ctx->device), ctx->supported_types);
                return false;
            }
            break;
        default:
            QNN_LOG_DEBUG("unsupported data type %s", type_name);
            return false;
    }

    return true;
}

bool ggml_qnn_supports_matmul_op(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op) {
    GGML_UNUSED(ctx);

    auto *src0 = op->src[0];
    auto *src1 = op->src[1];
    if (src0->type != src1->type || src0->type != op->type) {
        // current qnn implementation only supports the same type for src0 and src1
        QNN_LOG_DEBUG("src0 type %d and src1 type %d and op type %d are not equal", src0->type, src1->type, op->type);
        return false;
    }

    if ((src1->ne[2] % src0->ne[2]) != 0 || (src1->ne[3] % src0->ne[3]) != 0 || ctx->device == QNN_BACKEND_NPU) {
        /*
         * TODO: remove the blocker here when NPU backend supports mul_mat like this:
         *   [ne03, ne02, n, k] * [ne03 * x, ne02 * y, m, k] -> [ne03 * x, ne02 * y, m, n]
         */
        QNN_LOG_DEBUG("src0 and src1 dimensions are not equal");
        return false;
    }

    return true;
}

} // namespace

namespace qnn {

bool ggml_qnn_supports_op(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op) {
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (op->op == GGML_OP_UNARY) {
        const auto unary_op = ggml_get_unary_op(op);
        if (unary_op == GGML_UNARY_OP_GELU && ctx->device == QNN_BACKEND_NPU) {
            // TODO: fix this when NPU supports GELU
            QNN_LOG_DEBUG("unsupported unary op GGML_UNARY_OP_GELU for NPU");
            return false;
        }

        if (!kQnnUnaryOpsTable[kGgmlUnaryOpStart + unary_op]) {
            QNN_LOG_DEBUG("unsupported unary op %d", unary_op);
            return false;
        }

        if (!op->src[0]) {
            QNN_LOG_DEBUG("src0 is nullptr");
            return false;
        }
    } else {
        if (!kQnnUnaryOpsTable[op->op] && !kQnnBinaryOpsTable[op->op]) {
            QNN_LOG_DEBUG("[%s] unsupported op", ggml_op_name(op->op));
            return false;
        }

        auto *src0 = op->src[0];
        auto *src1 = op->src[1];
        if (!ggml_qnn_supports_tensor(ctx, src0) || !ggml_qnn_supports_tensor(ctx, op) ||
            (kQnnBinaryOpsTable[op->op] && !ggml_qnn_supports_tensor(ctx, src1))) {
            QNN_LOG_DEBUG("[%s] unsupported tensor", ggml_op_name(op->op));
            return false;
        }

        switch (op->op) {
            case GGML_OP_ADD:
                if (!is_tensor_dimensions_equal(src0, src1)) {
                    QNN_LOG_DEBUG("src0 and src1 dimensions are not equal");
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

bool ggml_qnn_forward(ggml_backend_qnn_device_context *ctx, struct ggml_tensor *tensor) {
    size_t unary_op_idx = tensor->op;
    if (tensor->op == GGML_OP_UNARY) {
        unary_op_idx = kGgmlUnaryOpStart + ggml_get_unary_op(tensor);
    }

    auto unary_op = kQnnUnaryOpsTable[unary_op_idx];
    if (unary_op) {
        return unary_op(ctx, tensor->src[0], tensor);
    }

    auto binary_op = kQnnBinaryOpsTable[tensor->op];
    if (binary_op) {
        return binary_op(ctx, tensor->src[0], tensor->src[1], tensor);
    }

    QNN_LOG_WARN("[forward]unsupported op %s", ggml_op_desc(tensor));
    return false;
}

} // namespace qnn
