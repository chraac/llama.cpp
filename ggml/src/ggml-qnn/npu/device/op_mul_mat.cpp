#include "op_mul_mat.hpp"

#include <HTP/core/intrinsics.h>

#include "quants.hpp"
#include "vtcm_mem.hpp"

namespace {

inline float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kElementsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;
    HVX_Vector   sum       = Q6_V_vzero();

    // TODO: prefetch or just use VTCM?
    while (iptr0 < iptr0_end) {
        HVX_Vector curr0 = *iptr0++;
        HVX_Vector curr1 = *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        sum              = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
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
        sum                      = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
        prev0                    = curr0;
        prev1                    = curr1;
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr0) > hexagon::kBytesPerVector) ? *iptr0 : prev0;
        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr1) > hexagon::kBytesPerVector) ? *iptr1 : prev1;
        curr1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_V_valign_VVR(Q6_Vqf32_vmpy_VsfVsf(curr0, curr1), Q6_V_vzero(), leftover_bytes), sum);
    }

    // TODO: do we have a better way to do the reduction?
    for (size_t i = kElementsPerVector / 2; i > 0; i /= 2) {
        sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vror_VR(sum, i * sizeof(float)));
    }

    float result;
    q6op_vstu_variable_ARV(&result, sizeof(float), Q6_Vsf_equals_Vqf32(sum));
    return result;
}

// TODO: merge with vec_dot_product_f32_f32?
inline float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(npu_device_fp16_t);
    constexpr const size_t kFloatsPerVector   = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kElementsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;
    HVX_Vector   sum_hi    = Q6_V_vzero();
    HVX_Vector   sum_lo    = Q6_V_vzero();

    // TODO: prefetch or just use VTCM?
    while (iptr0 < iptr0_end) {
        HVX_Vector     curr0  = *iptr0++;
        HVX_Vector     curr1  = *iptr1++;
        HVX_Vector     s0     = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector     s1     = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        HVX_VectorPair result = Q6_Wqf32_vmpy_VhfVhf(s0, s1);
        sum_hi                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(result), sum_hi);
        sum_lo                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(result), sum_lo);
        prev0                 = curr0;
        prev1                 = curr1;
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
        HVX_Vector     s0        = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector     s1        = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        HVX_VectorPair result    = Q6_Wqf32_vmpy_VhfVhf(s0, s1);
        sum_hi                   = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(result), sum_hi);
        sum_lo                   = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(result), sum_lo);
        prev0                    = curr0;
        prev1                    = curr1;
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(npu_device_fp16_t);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr0) > hexagon::kBytesPerVector) ? *iptr0 : prev0;
        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr1) > hexagon::kBytesPerVector) ? *iptr1 : prev1;
        curr1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        HVX_VectorPair result = Q6_Wqf32_vmpy_VhfVhf(curr0, curr1);

        // TODO: can we do this better?
        if (leftover > kFloatsPerVector) {
            sum_hi = Q6_Vqf32_vadd_Vqf32Vqf32(
                Q6_V_valign_VVR(Q6_V_hi_W(result), Q6_V_vzero(), (leftover % kFloatsPerVector) * sizeof(float)),
                sum_hi);
            sum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(result), sum_lo);
        } else {
            sum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(
                Q6_V_valign_VVR(Q6_V_lo_W(result), Q6_V_vzero(), leftover * sizeof(float)), sum_lo);
        }
    }

    // TODO: do we have a better way to do the reduction?
    sum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(sum_hi, sum_lo);
    for (size_t i = kFloatsPerVector / 2; i > 0; i /= 2) {
        sum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(sum_lo, Q6_V_vror_VR(sum_lo, i * sizeof(float)));
    }

    float result;
    q6op_vstu_variable_ARV(&result, sizeof(float), Q6_Vsf_equals_Vqf32(sum_lo));
    return result;
}

template <typename T> struct get_data_type {};

template <typename _TyData> struct get_data_type<float (*)(const _TyData *, const _TyData *, size_t)> {
    using type = _TyData;
};

template <auto _DotFunc>
void mul_mat_impl(hexagon::tensor * src0, hexagon::tensor * src1, hexagon::tensor * dst,
                  hexagon::compute_params * params) {
    using data_type                           = typename get_data_type<decltype(_DotFunc)>::type;
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(data_type);

    const auto   r02          = src1->get_ne(2) / src0->get_ne(2);
    const auto   r03          = src1->get_ne(3) / src0->get_ne(3);
    const auto * src0_ptr     = reinterpret_cast<const uint8_t *>(src0->get_data());
    const auto * src1_ptr     = reinterpret_cast<const uint8_t *>(src1->get_data());
    auto *       dst_ptr      = reinterpret_cast<uint8_t *>(dst->get_data());
    const auto   total_planes = dst->get_ne(3) * dst->get_ne(2);

    const auto start_end_plane = (total_planes >= params->tcnt) ?
                                     hexagon::get_thread_work_slice(total_planes, params->tidx, params->tcnt) :
                                     std::pair<int64_t, int64_t>{ 0, total_planes };
    // TODO: should we handle the case that dst->get_ne(1) < tcnt?
    const auto start_end_row   = (total_planes >= params->tcnt) ?
                                     std::pair<int64_t, int64_t>{ 0, dst->get_ne(1) } :
                                     hexagon::get_thread_work_slice(dst->get_ne(1), params->tidx, params->tcnt);
    if (start_end_row.second <= start_end_row.first) {
        return;
    }

    const bool is_quantized         = hexagon::is_quantized_type(src0->get_type());
    const auto src0_actual_row_size = hexagon::get_dequantized_row_size(src0);
    auto *     dequantize_row_func  = hexagon::get_type_traits(src0->get_type()).dequantize_row;
    if (is_quantized && dequantize_row_func == nullptr) {
        DEVICE_LOG_ERROR("Unsupported quantized src0 type: %d, dequantize_row_func is null\n", src0->get_type());
        return;
    }

    // cache the src0 plane in VTCM
    // TODO: should we skip the one plane matrix?
    const uint8_t * original_plane_ptr    = nullptr;
    size_t          src0_plane_cache_size = src0_actual_row_size * src0->get_ne(1);
    uint8_t *       src0_plane_cache_ptr  = params->get_cache(src0_plane_cache_size, is_quantized);
    DEVICE_LOG_DEBUG("mul_mat_impl src0_actual_row_size: %zu, is_quantized: %d, vtcm_mem: %p(%zu)\n",
                     src0_actual_row_size, is_quantized, (void *) src0_plane_cache_ptr, src0_plane_cache_size);
    for (int64_t ip = start_end_plane.first; ip < start_end_plane.second; ip++) {
        const auto   i3         = ip / dst->get_ne(2);
        const auto   i2         = ip - i3 * dst->get_ne(2);
        const auto * src0_plane = src0_ptr + i3 / r03 * src0->get_nb(3) + i2 / r02 * src0->get_nb(2);
        const auto * src1_plane = src1_ptr + i3 * src1->get_nb(3) + i2 * src1->get_nb(2);
        auto *       dst_plane  = dst_ptr + i3 * dst->get_nb(3) + i2 * dst->get_nb(2);

        if (src0_plane_cache_ptr) {
            if (original_plane_ptr != src0_plane) {
                if (is_quantized) {
                    for (int64_t ir = 0; ir < src0->get_ne(1); ir++) {
                        auto * src0_row = src0_plane + ir * src0->get_nb(1);
                        auto * dst_row  = reinterpret_cast<float *>(src0_plane_cache_ptr + ir * src0_actual_row_size);
                        dequantize_row_func(src0_row, reinterpret_cast<float *>(dst_row), src0->get_ne(0),
                                            params->f16_to_f32_table);
                    }
                } else {
                    memcpy(src0_plane_cache_ptr, src0_plane, src0_plane_cache_size);
                }
                original_plane_ptr = src0_plane;
            }

            src0_plane = src0_plane_cache_ptr;
        }

        for (int64_t i1 = start_end_row.first; i1 < start_end_row.second; i1++) {
            auto * src1_row = src1_plane + i1 * src1->get_nb(1);
            auto * dst_row  = reinterpret_cast<float *>(dst_plane + i1 * dst->get_nb(1));
            for (int64_t i0 = 0; i0 < dst->get_ne(0); i0++) {
                auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                if (i0 + 1 < dst->get_ne(0)) {
                    if (!src0_plane_cache_ptr) {  // TODO: should we use small kL2FetchAheadVectors?
                        int32_t l2fetch_vectors =
                            Q6_R_min_RR(src0->get_ne(1) / kElementsPerVector, hexagon::kL2FetchAheadVectors);
                        hexagon::l2fetch(src0_row + src0_actual_row_size, hexagon::kBytesPerVector,
                                         hexagon::kBytesPerVector, l2fetch_vectors, 0);
                    }
                } else if (ip + 1 < start_end_plane.second) {
                    // TODO: should we use small kL2FetchAheadVectors?
                    int32_t l2fetch_vectors =
                        Q6_R_min_RR(src1->get_ne(1) / kElementsPerVector, hexagon::kL2FetchAheadVectors);
                    hexagon::l2fetch(src1_row + src1->get_nb(1), hexagon::kBytesPerVector, hexagon::kBytesPerVector,
                                     l2fetch_vectors, 0);
                }

                // TODO: figure dst how to handle a entire row
                *dst_row++ = _DotFunc(reinterpret_cast<const data_type *>(src0_row),
                                      reinterpret_cast<const data_type *>(src1_row), (size_t) src0->get_ne(0));
            }
        }
    }
}

}  // namespace

namespace hexagon {

bool mul_mat_f32(hexagon::tensor * out, compute_params * params) {
    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "mul_mat_f32 requires max dims 4");
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    // TODO: array?
    switch (src1->get_type()) {
        case NPU_DATA_TYPE_F32:
            mul_mat_impl<vec_dot_product_f32_f32>(src0, src1, out, params);
            return true;

        case NPU_DATA_TYPE_F16:
            mul_mat_impl<vec_dot_product_f16_f16>(src0, src1, out, params);
            return true;
        default:
            break;
    }

    DEVICE_LOG_ERROR("Unsupported src1 tensor type: %d\n", src1->get_type());
    return false;
}

bool is_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                          const npu_device_tensor_spec & dst, npu_device_tensor_op op) {
    if (op != NPU_OP_MUL_MAT) {
        DEVICE_LOG_DEBUG("op is not MUL_MAT: %d\n", op);
        return false;
    }

    if (dst.type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst.type));
        return false;
    }

    if (src0.type != src1.type) {
        if (src1.type != NPU_DATA_TYPE_F32) {
            DEVICE_LOG_DEBUG("[%s]src0.type(%s) and src1.type(%s) mismatch and src1 is not F32\n", op_get_name(op),
                             get_type_name(src0.type), get_type_name(src1.type));
            return false;
        }

        const auto type_traits = get_type_traits(src0.type);
        if (!type_traits.is_quantized || type_traits.dequantize_row == nullptr) {
            DEVICE_LOG_DEBUG("[%s]src0.type(%s) and src1.type(%s) mismatch and src0 is not quantized\n",
                             op_get_name(op), get_type_name(src0.type), get_type_name(src1.type));
            return false;
        }

        if (src0.ne[0] % type_traits.blck_size) {
            DEVICE_LOG_DEBUG("[%s]src0.type(%s) ne[0] is not aligned: %ld\n", op_get_name(op), get_type_name(src0.type),
                             (long) src0.ne[0]);
            return false;
        }

        DEVICE_LOG_DEBUG("[%s]supported quantized src0.type(%s) and src1.type(%s)\n", op_get_name(op),
                         get_type_name(src0.type), get_type_name(src1.type));
    }

    if (src0.ne[0] != src1.ne[0] || src0.ne[1] != dst.ne[0]) {
        DEVICE_LOG_DEBUG("[%s]src0 and src1 cannot multiply: %ldx%ld vs %ldx%ld\n", op_get_name(op), (long) src0.ne[0],
                         (long) src0.ne[1], (long) src1.ne[0], (long) src1.ne[1]);
        return false;
    }

    if (src1.ne[1] != dst.ne[1] || src1.ne[2] != dst.ne[2] || src1.ne[3] != dst.ne[3]) {
        DEVICE_LOG_DEBUG("[%s]src1 and dst dimensions not match: %ldx%ld vs %ldx%ld\n", op_get_name(op),
                         (long) src1.ne[2], (long) src1.ne[3], (long) dst.ne[2], (long) dst.ne[3]);
        return false;
    }

    if (src1.ne[2] % src0.ne[2] || src1.ne[3] % src0.ne[3]) {
        DEVICE_LOG_DEBUG("[%s]src0 cannot broadcast to src1: %ldx%ld vs %ldx%ld\n", op_get_name(op), (long) src0.ne[2],
                         (long) src0.ne[3], (long) src1.ne[2], (long) src1.ne[3]);
        return false;
    }

    return true;
}

}  // namespace hexagon
