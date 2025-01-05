
#include "op-config.hpp"

namespace {

using op_dims_calc_func_t = void (*)(const std::vector<const qnn::ggml_dimension_array_t> &input_dims,
                                     qnn::ggml_dimension_array_t &output_dims);

void element_wise_op_dims(const std::vector<const qnn::ggml_dimension_array_t> &input_dims,
                          qnn::ggml_dimension_array_t &output_dims) {
    for (size_t i = 1; i < std::size(output_dims); i++) {
        output_dims[i] = input_dims.front()[i];
    }
}

void mat_mul_op_dims(const std::vector<const qnn::ggml_dimension_array_t> &input_dims,
                     qnn::ggml_dimension_array_t &output_dims) {
    GGML_ASSERT(input_dims.size() == 2);
    output_dims[0] = input_dims.front()[1];
    output_dims[1] = input_dims.back()[1];
}

struct qnn_op_caps_t {
    const char *qnn_op_name = nullptr;
    const size_t input_param_count = 0;
    op_dims_calc_func_t calc_dims_func = nullptr;
};

constexpr const qnn_op_caps_t kOpCaps[] = {
    {}, // GGML_OP_NONE
    {}, // GGML_OP_DUP
    {
        // GGML_OP_ADD
        QNN_OP_ELEMENT_WISE_ADD, // qnn_op_name
        2,                       // input_param_count
        element_wise_op_dims,    // calc_dims_func
    },
    {}, // GGML_OP_ADD1
    {}, // GGML_OP_ACC
    {
        // GGML_OP_SUB
        QNN_OP_ELEMENT_WISE_SUBTRACT, // qnn_op_name
        2,                            // input_param_count
        element_wise_op_dims,         // calc_dims_func
    },
    {
        // GGML_OP_MUL
        QNN_OP_ELEMENT_WISE_MULTIPLY, // qnn_op_name
        2,                            // input_param_count
        element_wise_op_dims,         // calc_dims_func
    },
    {
        // GGML_OP_DIV
        QNN_OP_ELEMENT_WISE_DIVIDE, // qnn_op_name
        2,                          // input_param_count
        element_wise_op_dims,       // calc_dims_func
    },
    {}, // GGML_OP_SQR
    {
        // GGML_OP_SQRT
        QNN_OP_ELEMENT_WISE_SQUARE_ROOT, // qnn_op_name
        1,                               // input_param_count
        element_wise_op_dims,            // calc_dims_func
    },
    {
        // GGML_OP_LOG
        QNN_OP_ELEMENT_WISE_LOG, // qnn_op_name
        1,                       // input_param_count
        element_wise_op_dims,    // calc_dims_func
    },
    {}, // GGML_OP_SIN
    {}, // GGML_OP_COS
    {}, // GGML_OP_SUM
    {}, // GGML_OP_SUM_ROWS
    {}, // GGML_OP_MEAN
    {}, // GGML_OP_ARGMAX
    {}, // GGML_OP_COUNT_EQUAL
    {}, // GGML_OP_REPEAT
    {}, // GGML_OP_REPEAT_BACK
    {}, // GGML_OP_CONCAT
    {}, // GGML_OP_SILU_BACK
    {}, // GGML_OP_NORM
    {}, // GGML_OP_RMS_NORM
    {}, // GGML_OP_RMS_NORM_BACK
    {}, // GGML_OP_GROUP_NORM
    {
        // GGML_OP_MUL_MAT
        QNN_OP_MAT_MUL,  // qnn_op_name
        2,               // input_param_count
        mat_mul_op_dims, // calc_dims_func
    },
    {}, // GGML_OP_MUL_MAT_ID
    {}, // GGML_OP_OUT_PROD
    {}, // GGML_OP_SCALE
    {}, // GGML_OP_SET
    {}, // GGML_OP_CPY
    {}, // GGML_OP_CONT
    {
        // GGML_OP_RESHAPE
        QNN_OP_RESHAPE, // qnn_op_name
        1,              // input_param_count
        nullptr,        // TODO: calc_dims_func
    },
    {}, // GGML_OP_VIEW
    {}, // GGML_OP_PERMUTE
    {}, // GGML_OP_TRANSPOSE
    {}, // GGML_OP_GET_ROWS
    {}, // GGML_OP_GET_ROWS_BACK
    {}, // GGML_OP_DIAG
    {}, // GGML_OP_DIAG_MASK_INF
    {}, // GGML_OP_DIAG_MASK_ZERO
    {}, // GGML_OP_SOFT_MAX
    {}, // GGML_OP_SOFT_MAX_BACK
    {}, // GGML_OP_ROPE
    {}, // GGML_OP_ROPE_BACK
    {}, // GGML_OP_CLAMP
    {}, // GGML_OP_CONV_TRANSPOSE_1D
    {}, // GGML_OP_IM2COL
    {}, // GGML_OP_IM2COL_BACK
    {}, // GGML_OP_CONV_TRANSPOSE_2D
    {}, // GGML_OP_POOL_1D
    {}, // GGML_OP_POOL_2D
    {}, // GGML_OP_POOL_2D_BACK
    {}, // GGML_OP_UPSCALE
    {}, // GGML_OP_PAD
    {}, // GGML_OP_PAD_REFLECT_1D
    {}, // GGML_OP_ARANGE

    {}, // GGML_OP_TIMESTEP_EMBEDDING
    {}, // GGML_OP_ARGSORT
    {}, // GGML_OP_LEAKY_RELU

    {}, // GGML_OP_FLASH_ATTN_EXT
    {}, // GGML_OP_FLASH_ATTN_BACK
    {}, // GGML_OP_SSM_CONV
    {}, // GGML_OP_SSM_SCAN
    {}, // GGML_OP_WIN_PART
    {}, // GGML_OP_WIN_UNPART
    {}, // GGML_OP_GET_REL_POS
    {}, // GGML_OP_ADD_REL_POS
    {}, // GGML_OP_RWKV_WKV6

    {}, // GGML_OP_UNARY

    {}, // GGML_OP_MAP_UNARY
    {}, // GGML_OP_MAP_BINARY

    {}, // GGML_OP_MAP_CUSTOM1_F32
    {}, // GGML_OP_MAP_CUSTOM2_F32
    {}, // GGML_OP_MAP_CUSTOM3_F32

    {}, // GGML_OP_MAP_CUSTOM1
    {}, // GGML_OP_MAP_CUSTOM2
    {}, // GGML_OP_MAP_CUSTOM3

    {}, // GGML_OP_CROSS_ENTROPY_LOSS
    {}, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    {}, // GGML_OP_OPT_STEP_ADAMW

    // ggml_unary_op
    {}, // GGML_UNARY_OP_ABS
    {}, // GGML_UNARY_OP_SGN
    {}, // GGML_UNARY_OP_NEG
    {}, // GGML_UNARY_OP_STEP
    {}, // GGML_UNARY_OP_TANH
    {}, // GGML_UNARY_OP_ELU
    {}, // GGML_UNARY_OP_RELU
    {}, // GGML_UNARY_OP_SIGMOID
    {
        // GGML_UNARY_OP_GELU
        QNN_OP_GELU, // qnn_op_name
        1,           // input_param_count
        nullptr,     // TODO: calc_dims_func
    },
    {}, // GGML_UNARY_OP_GELU_QUICK
    {}, // GGML_UNARY_OP_SILU
    {}, // GGML_UNARY_OP_HARDSWISH
    {}, // GGML_UNARY_OP_HARDSIGMOID
    {}, // GGML_UNARY_OP_EXP
};

static_assert(kOpCaps[GGML_OP_NONE].calc_dims_func == nullptr, "GGML_OP_NONE should not have calc_dims_func function");
static_assert(kOpCaps[GGML_OP_ADD].calc_dims_func == element_wise_op_dims,
              "GGML_OP_ADD does not have element_wise_op_dims function");
static_assert(kOpCaps[GGML_OP_MUL_MAT].calc_dims_func == mat_mul_op_dims,
              "GGML_OP_ADD does not have element_wise_op_dims function");
static_assert(kOpCaps[GGML_OP_LOG].calc_dims_func == element_wise_op_dims,
              "GGML_OP_LOG does not have element_wise_op_dims function");
static_assert(std::size(kOpCaps) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kOpCaps table");

} // namespace

namespace qnn {

size_t get_qnn_op_index(const ggml_tensor *tensor) {
    if (tensor->op == GGML_OP_UNARY) {
        return kGgmlUnaryOpStart + ggml_get_unary_op(tensor);
    }

    return tensor->op;
}

void get_ggml_op_output_dimensions(const std::vector<const ggml_dimension_array_t> &input_dims, size_t op,
                                   ggml_dimension_array_t &output_dims) {
    GGML_ASSERT(op < std::size(kOpCaps));
    auto get_dims = kOpCaps[op].calc_dims_func;
    GGML_ASSERT(get_dims);
    get_dims(input_dims, output_dims);
}

const char *get_qnn_op_name(size_t op) {
    GGML_ASSERT(op < std::size(kOpCaps));
    GGML_ASSERT(kOpCaps[op].qnn_op_name);
    return kOpCaps[op].qnn_op_name;
}

size_t get_qnn_op_input_param_count(size_t op) {
    GGML_ASSERT(op < std::size(kOpCaps));
    return kOpCaps[op].input_param_count;
}

} // namespace qnn
