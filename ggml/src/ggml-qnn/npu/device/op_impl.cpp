

#include "op_impl.hpp"

#include <hexagon_types.h>
#include <HTP/core/intrinsics.h>

#include <type_traits>

#include "op_flash_attn.hpp"
#include "op_mul_mat.hpp"
#include "quants.hpp"
#include "vec_dot.hpp"

namespace {

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector), typename _TyData>
inline void vec_op_impl(const _TyData * src0, const _TyData * src1, size_t count, _TyData * dst) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TyData);

    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kElementsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector * optr      = ((HVX_Vector *) dst);  // framework will ensure the dst is aligned
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;

    while (iptr0 < iptr0_end) {
        HVX_Vector curr0 = *iptr0++;
        HVX_Vector curr1 = *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++          = _OpIntrinsic(s0, s1);
        prev0            = curr0;
        prev1            = curr1;
    }

    if ((iptr0_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool       iptr0_aligned = hexagon::is_addr_aligned(iptr0);
        HVX_Vector curr0         = iptr0_aligned ? prev0 : *iptr0;
        iptr0                    = iptr0_aligned ? iptr0 : iptr0 + 1;
        bool       iptr1_aligned = hexagon::is_addr_aligned(iptr1);
        HVX_Vector curr1         = iptr1_aligned ? prev1 : *iptr1;
        iptr1                    = iptr1_aligned ? iptr1 : iptr1 + 1;
        HVX_Vector s0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++                  = _OpIntrinsic(s0, s1);
        prev0                    = curr0;
        prev1                    = curr1;
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(_TyData);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr0) > hexagon::kBytesPerVector) ? *iptr0 : prev0;
        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr1) > hexagon::kBytesPerVector) ? *iptr1 : prev1;
        curr1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        q6op_vstu_variable_ARV(optr, leftover_bytes, _OpIntrinsic(curr0, curr1));
    }
}

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector)>
inline void vec_op_f32_f32(const float * src0, const float * src1, size_t count, float * dst) {
    vec_op_impl<_OpIntrinsic, float>(src0, src1, count, dst);
}

inline HVX_Vector vadd_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

inline HVX_Vector vsub_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

inline HVX_Vector vmul_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector)>
inline void vec_op_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count,
                           npu_device_fp16_t * dst) {
    vec_op_impl<_OpIntrinsic, npu_device_fp16_t>(src0, src1, count, dst);
}

inline HVX_Vector vadd_f16_f16(HVX_Vector a, HVX_Vector b) {
    // TODO: fix this since qf16 has less precision than fp16
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
}

inline HVX_Vector vsub_f16_f16(HVX_Vector a, HVX_Vector b) {
    // TODO: fix this since qf16 has less precision than fp16
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
}

inline HVX_Vector vmul_f16_f16(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

template <typename T> struct get_data_type {};

template <typename _TyData> struct get_data_type<void (*)(const _TyData *, const _TyData *, size_t, _TyData *)> {
    using type = _TyData;
};

template <typename _TyData, typename _TyParam>
struct get_data_type<void (*)(const _TyData *, size_t, _TyParam, _TyData *)> {
    using type       = _TyData;
    using param_type = typename std::remove_cv<typename std::remove_reference<_TyData>::type>::type;
};

template <auto _RowFunc> bool element_wise_op(hexagon::tensor * out, hexagon::compute_params * params) {
    using data_type = typename get_data_type<decltype(_RowFunc)>::type;

    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    if (src0->get_ne(0) != src1->get_ne(0)) {
        // TODO: handle this case
        DEVICE_LOG_ERROR("src0[0] and src1[0] not match: %ld vs %ld\n", (long) src0->get_ne(0), (long) src1->get_ne(0));
        return false;
    }

    const auto * src0_ptr      = reinterpret_cast<const uint8_t *>(src0->get_read_buffer());
    const auto * src1_ptr      = reinterpret_cast<const uint8_t *>(src1->get_read_buffer());
    auto *       dst_ptr       = reinterpret_cast<uint8_t *>(out->get_write_buffer());
    auto         total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto   rows_per_cube = out->get_ne(2) * out->get_ne(1);
    const auto   start_end     = hexagon::get_thread_work_slice(total_rows, params->tidx, params->tcnt);

    if (start_end.first >= start_end.second) {
        return true;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->tidx);

    const size_t valid_row_bytes = src0->get_ne(0) * sizeof(data_type);
    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03 = ir / rows_per_cube;
        const auto i02 = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?
        const auto i13 = i03 % src1->get_ne(3);
        const auto i12 = i02 % src1->get_ne(2);
        const auto i11 = i01 % src1->get_ne(1);

        auto * src1_plane = src1_ptr + i13 * src1->get_nb(3) + i12 * src1->get_nb(2);
        auto * src0_row   = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * src1_row   = src1_plane + i11 * src1->get_nb(1);
        auto * dst_row    = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        if (ir + 1 < start_end.second) {
            hexagon::l2fetch_row(src0_row + src0->get_nb(1), valid_row_bytes);
            hexagon::l2fetch_row(src1_row + src1->get_nb(1), valid_row_bytes);
        }

        _RowFunc(reinterpret_cast<const data_type *>(src0_row), reinterpret_cast<const data_type *>(src1_row),
                 static_cast<size_t>(out->get_ne(0)), reinterpret_cast<data_type *>(dst_row));
    }

    return true;
}

bool is_same_shape(const npu_device_tensor_spec & src, const npu_device_tensor_spec & dst) {
    for (size_t i = 0; i < DEVICE_TENSOR_MAX_DIMS; ++i) {
        if (src.ne[i] != dst.ne[i]) {
            return false;
        }
    }

    return true;
}

bool is_element_wise_op_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                                  const npu_device_tensor_spec * srcs, size_t src_len) {
    if (op != NPU_OP_ADD && op != NPU_OP_SUB && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (!dst || !srcs || src_len < 2) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    const auto & src0 = srcs[0];
    const auto & src1 = srcs[1];
    if (dst->type != src0.type || dst->type != src1.type) {
        DEVICE_LOG_DEBUG("[%s]src0.type and dst.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(src0.type), hexagon::get_type_name(dst->type));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32 && dst->type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(dst->type));
        return false;
    }

    // TODO: fix FP16 add/sub
    if (dst->type == NPU_DATA_TYPE_F16 && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(dst->type));
        return false;
    }

    if (src0.ne[0] != src1.ne[0]) {
        DEVICE_LOG_DEBUG("[%s]src0.ne[0] and src1.ne[0] not match: %ld vs %ld\n", hexagon::op_get_name(op),
                         (long) src0.ne[0], (long) src1.ne[0]);
        return false;
    }

    if (!is_same_shape(src0, *dst)) {
        DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape\n", hexagon::op_get_name(op));
        return false;
    }

    return true;
}

void rms_norm_vec_f32(const float * src, size_t count, float eps, float * dst) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector * src_vec_ptr = ((HVX_Vector *) src);
    HVX_Vector * src_vec_end = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_Vector   prev        = *src_vec_ptr++;
    HVX_Vector   sum         = Q6_V_vzero();
    while (src_vec_ptr < src_vec_end) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        sum             = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(s0, s0));
        prev            = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       src_ptr_aligned = hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr            = src_ptr_aligned ? prev : *src_vec_ptr;
        src_vec_ptr                = src_ptr_aligned ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector s0              = Q6_V_valign_VVR(curr, prev, (size_t) src);
        sum                        = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(s0, s0));
        prev                       = curr;
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        sum  = Q6_Vqf32_vadd_Vqf32Vqf32(sum,
                                        Q6_V_valign_VVR(Q6_Vqf32_vmpy_VsfVsf(curr, curr), Q6_V_vzero(), leftover_bytes));
    }

    const float mean  = hexagon::vec_reduction_f32(sum) / count;  // TODO: figure out how to do division in vector
    const float scale = 1.0f / sqrtf(mean + eps);                 // TODO: use buildin blas sqrtf?

    HVX_Vector scale_vec     = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));
    src_vec_ptr              = ((HVX_Vector *) src);
    prev                     = *src_vec_ptr++;
    HVX_Vector * dst_vec_ptr = ((HVX_Vector *) dst);  // framework will ensure the dst is aligned
    while (src_vec_ptr < src_vec_end) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        *dst_vec_ptr++  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec));
        prev            = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       src_ptr_aligned = hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr            = src_ptr_aligned ? prev : *src_vec_ptr;
        src_vec_ptr                = src_ptr_aligned ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector s0              = Q6_V_valign_VVR(curr, prev, (size_t) src);
        *dst_vec_ptr++             = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec));
        prev                       = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(curr, scale_vec)));
    }
}

// TODO: merge with element_wise_op?
template <auto _RowFunc> bool unary_op(hexagon::tensor * out, hexagon::compute_params * params) {
    using data_type  = typename get_data_type<decltype(_RowFunc)>::type;
    using param_type = typename get_data_type<decltype(_RowFunc)>::param_type;

    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");
    auto * src0 = out->get_src(0);
    if (!src0) {
        return true;  // skip if no src
    }

    const auto * src0_ptr      = reinterpret_cast<const uint8_t *>(src0->get_read_buffer());
    auto *       dst_ptr       = reinterpret_cast<uint8_t *>(out->get_write_buffer());
    auto         total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto   rows_per_cube = out->get_ne(2) * out->get_ne(1);
    const auto   start_end     = hexagon::get_thread_work_slice(total_rows, params->tidx, params->tcnt);
    if (start_end.first >= start_end.second) {
        return true;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER(out, params->tidx);

    const auto   param           = out->get_op_param<param_type>(0);
    const size_t valid_row_bytes = src0->get_ne(0) * sizeof(data_type);
    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03 = ir / rows_per_cube;
        const auto i02 = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?

        auto * src0_row = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * dst_row  = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        if (ir + 1 < start_end.second) {
            hexagon::l2fetch_row(src0_row + src0->get_nb(1), valid_row_bytes);
        }

        _RowFunc(reinterpret_cast<const data_type *>(src0_row), static_cast<size_t>(out->get_ne(0)), param,
                 reinterpret_cast<data_type *>(dst_row));
    }

    return true;
}

bool is_unary_op_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                           const npu_device_tensor_spec * srcs, size_t src_len) {
    if (op != NPU_OP_RMS_NORM) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (!dst || !srcs || src_len < 1) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    const auto & src0 = srcs[0];
    if (dst->type != src0.type) {
        DEVICE_LOG_DEBUG("[%s]src0.type and dst.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(src0.type), hexagon::get_type_name(dst->type));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(dst->type));
        return false;
    }

    if (!is_same_shape(src0, *dst)) {
        DEVICE_LOG_DEBUG("[%s]src0 and dst have different shape\n", hexagon::op_get_name(op));
        return false;
    }

    return true;
}

struct op_capabilities {
    npu_device_tensor_op               op;
    hexagon::op_is_supported_func_type is_supported;
    hexagon::compute_func_type         compute_funcs[NPU_DATA_TYPE_COUNT];
    bool                               requires_thread_barrier = false;
};

constexpr const op_capabilities kOpCapabilities[] = {
    {
     NPU_OP_MUL_MAT, hexagon::is_mul_mat_supported,
     {
            hexagon::mul_mat_f32,  // NPU_DATA_TYPE_F32
            nullptr,               // NPU_DATA_TYPE_F16
        },       true,
     },
    {
     NPU_OP_ADD,              is_element_wise_op_supported,
     {
            element_wise_op<vec_op_f32_f32<vadd_f32_f32>>,  // NPU_DATA_TYPE_F32
            element_wise_op<vec_op_f16_f16<vadd_f16_f16>>,  // NPU_DATA_TYPE_F16
        }, false,
     },
    {
     NPU_OP_SUB,           is_element_wise_op_supported,
     {
            element_wise_op<vec_op_f32_f32<vsub_f32_f32>>,  // NPU_DATA_TYPE_F32
            element_wise_op<vec_op_f16_f16<vsub_f16_f16>>,  // NPU_DATA_TYPE_F16
        },          false,
     },
    {
     NPU_OP_MUL,     is_element_wise_op_supported,
     {
            element_wise_op<vec_op_f32_f32<vmul_f32_f32>>,  // NPU_DATA_TYPE_F32
            element_wise_op<vec_op_f16_f16<vmul_f16_f16>>,  // NPU_DATA_TYPE_F16
        },       false,
     },
    {
     NPU_OP_RMS_NORM,              is_unary_op_supported,
     {
            unary_op<rms_norm_vec_f32>,  // NPU_DATA_TYPE_F32
            nullptr,                     // NPU_DATA_TYPE_F16
        }, false,
     },
    {
     NPU_OP_FLASH_ATTN,           hexagon::is_flash_attn_supported,
     {
            hexagon::flash_attn_f32,  // NPU_DATA_TYPE_F32
            nullptr,                  // NPU_DATA_TYPE_F16
        },          false,
     },
};

static_assert(kOpCapabilities[NPU_OP_MUL_MAT].compute_funcs[NPU_DATA_TYPE_F32] == hexagon::mul_mat_f32,
              "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert(std::size(kOpCapabilities) == NPU_OP_COUNT);
static_assert(kOpCapabilities[NPU_OP_MUL_MAT].op == NPU_OP_MUL_MAT, "kOpArray[NPU_OP_MUL_MAT].op != NPU_OP_MUL_MAT");
static_assert(kOpCapabilities[NPU_OP_MUL].op == NPU_OP_MUL, "kOpArray[NPU_OP_MUL].op != NPU_OP_MUL");
static_assert(kOpCapabilities[NPU_OP_RMS_NORM].op == NPU_OP_RMS_NORM,
              "kOpArray[NPU_OP_RMS_NORM].op != NPU_OP_RMS_NORM");
static_assert(kOpCapabilities[NPU_OP_FLASH_ATTN].op == NPU_OP_FLASH_ATTN,
              "kOpArray[NPU_OP_FLASH_ATTN].op != NPU_OP_FLASH_ATTN");

hexagon::compute_func_type get_compute_func_impl(npu_device_tensor_op op, npu_device_tensor_data_type type) {
    if (op >= NPU_OP_COUNT) {
        return nullptr;
    }

    return kOpCapabilities[op].compute_funcs[type];
}

}  // namespace

namespace hexagon {

compute_func_type get_compute_func(tensor * dst) {
    return get_compute_func_impl(dst->get_op(), dst->get_type());
}

bool requires_thread_barrier(npu_device_tensor_op op) {
    if (op >= NPU_OP_COUNT) {
        return false;
    }

    return kOpCapabilities[op].requires_thread_barrier;
}

bool support_op(npu_device_tensor_op op, const npu_device_tensor_spec * dst, const npu_device_tensor_spec * srcs,
                size_t src_len) {
    if (get_compute_func_impl(op, dst->type) == nullptr) {
        DEVICE_LOG_ERROR("[%s]unsupported, get_compute_func failed\n", op_get_name(op));
        return false;
    }

    auto is_supported_func = kOpCapabilities[op].is_supported;
    if (!is_supported_func || !is_supported_func(op, dst, srcs, src_len)) {
        DEVICE_LOG_DEBUG("[%s]unsupported, is_supported_func failed\n", op_get_name(op));
        return false;
    }

    return true;
}

}  // namespace hexagon
