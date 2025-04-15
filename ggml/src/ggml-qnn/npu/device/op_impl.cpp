

#include "op_impl.hpp"

#include <hexagon_types.h>
#include <HTP/core/intrinsics.h>

#include "tensor.hpp"

namespace {

constexpr const size_t kBytesPerVector  = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kFloatsPerVector = kBytesPerVector / sizeof(float);
constexpr const size_t kAlignMask       = kBytesPerVector - 1;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(void * addr) {
    return unaligned_bytes(addr) == 0;
}

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector)>
inline void vec_op_f32_f32(const float * src0, const float * src1, size_t count, float * dst) {
    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kFloatsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector * optr      = ((HVX_Vector *) dst);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;

    // TODO: prefetch?
    while (iptr0 < iptr0_end) {
        HVX_Vector curr0 = *iptr0++;
        HVX_Vector curr1 = *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++          = Q6_Vsf_equals_Vqf32(_OpIntrinsic(s0, s1));
        prev0            = curr0;
        prev1            = curr1;
    }

    if ((iptr0_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        HVX_Vector curr0 = is_addr_aligned(iptr0) ? prev0 : *iptr0++;
        HVX_Vector curr1 = is_addr_aligned(iptr1) ? prev1 : *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++          = Q6_Vsf_equals_Vqf32(_OpIntrinsic(s0, s1));
        prev0            = curr0;
        prev1            = curr1;
    }

    const size_t leftover       = count % kFloatsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 = (leftover_bytes + unaligned_bytes(iptr0) > kBytesPerVector) ? *iptr0 : prev0;
        curr0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + unaligned_bytes(iptr1) > kBytesPerVector) ? *iptr1 : prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        q6op_vstu_variable_ARV(optr, leftover_bytes, Q6_Vsf_equals_Vqf32(_OpIntrinsic(curr0, curr1)));
    }
}

inline HVX_Vector vadd_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vadd_VsfVsf(a, b);
}

inline HVX_Vector vsub_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vsub_VsfVsf(a, b);
}

inline HVX_Vector vmul_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vmpy_VsfVsf(a, b);
}

inline float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count) {
    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kFloatsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;
    HVX_Vector   sum       = Q6_V_vzero();

    // TODO: prefetch?
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
        HVX_Vector curr0 = is_addr_aligned(iptr0) ? prev0 : *iptr0++;
        HVX_Vector curr1 = is_addr_aligned(iptr1) ? prev1 : *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        sum              = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
        prev0            = curr0;
        prev1            = curr1;
    }

    const size_t leftover       = count % kFloatsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 = (leftover_bytes + unaligned_bytes(iptr0) > kBytesPerVector) ? *iptr0 : prev0;
        curr0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + unaligned_bytes(iptr1) > kBytesPerVector) ? *iptr1 : prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_V_valign_VVR(Q6_Vqf32_vmpy_VsfVsf(curr0, curr1), Q6_V_vzero(), leftover_bytes), sum);
    }

    // TODO: do we have a better way to do the reduction?
    for (size_t i = kFloatsPerVector / 2; i > 0; i /= 2) {
        sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vror_VR(sum, i * sizeof(float)));
    }

    float result;
    q6op_vstu_variable_ARV(&result, sizeof(float), Q6_Vsf_equals_Vqf32(sum));
    return result;
}

bool mul_mat_f32(hexagon::tensor * out) {
    if (!out) {
        return false;
    }

    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "mul_mat_f32 requires max dims 4");

    const auto   r02      = src1->get_ne(2) / src0->get_ne(2);
    const auto   r03      = src1->get_ne(3) / src0->get_ne(3);
    const auto * src0_ptr = reinterpret_cast<const uint8_t *>(src0->get_data());
    const auto * src1_ptr = reinterpret_cast<const uint8_t *>(src1->get_data());
    auto *       dst_ptr  = reinterpret_cast<uint8_t *>(out->get_data());
    for (int64_t i3 = 0; i3 < out->get_ne(3); i3++) {
        const auto * src0_cube = src0_ptr + i3 / r03 * src0->get_nb(3);
        const auto * src1_cube = src1_ptr + i3 * src1->get_nb(3);
        auto *       dst_cube  = dst_ptr + i3 * out->get_nb(3);
        for (int64_t i2 = 0; i2 < out->get_ne(2); i2++) {
            const auto * src0_plane = src0_cube + i2 / r02 * src0->get_nb(2);
            const auto * src1_plane = src1_cube + i2 * src1->get_nb(2);
            auto *       dst_plane  = dst_cube + i2 * out->get_nb(2);
            for (int64_t i1 = 0; i1 < out->get_ne(1); i1++) {
                // TODO: prefetch row?
                auto * src1_row = src1_plane + i1 * src1->get_nb(1);
                auto * dst_row  = reinterpret_cast<float *>(dst_plane + i1 * out->get_nb(1));
                for (int64_t i0 = 0; i0 < out->get_ne(0); i0++) {
                    auto * src0_row = src0_plane + i0 * src0->get_nb(1);
                    // TODO: figure out how to handle a entire row
                    *dst_row++ =
                        vec_dot_product_f32_f32(reinterpret_cast<const float *>(src0_row),
                                                reinterpret_cast<const float *>(src1_row), (size_t) src0->get_ne(0));
                }
            }
        }
    }

    return true;
}

template <typename _TySrc, typename _TyDst, void (*_RowFunc)(const _TySrc *, const _TySrc *, size_t, _TyDst *)>
bool element_wise_op(hexagon::tensor * out) {
    if (!out) {
        return false;
    }

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

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");

    const auto * src0_ptr = reinterpret_cast<const uint8_t *>(src0->get_data());
    const auto * src1_ptr = reinterpret_cast<const uint8_t *>(src1->get_data());
    auto *       dst_ptr  = reinterpret_cast<uint8_t *>(out->get_data());
    const auto   r13      = out->get_ne(3) / src1->get_ne(3);
    const auto   r12      = out->get_ne(2) / src1->get_ne(2);
    const auto   r11      = out->get_ne(1) / src1->get_ne(1);
    for (int64_t i3 = 0; i3 < out->get_ne(3); i3++) {
        const auto * src0_cube = src0_ptr + i3 * src0->get_nb(3);
        const auto * src1_cube = src1_ptr + i3 / r13 * src1->get_nb(3);
        auto *       dst_cube  = dst_ptr + i3 * out->get_nb(3);
        for (int64_t i2 = 0; i2 < out->get_ne(2); i2++) {
            const auto * src0_plane = src0_cube + i2 * src0->get_nb(2);
            const auto * src1_plane = src1_cube + i2 / r12 * src1->get_nb(2);
            auto *       dst_plane  = dst_cube + i2 * out->get_nb(2);
            for (int64_t i1 = 0; i1 < out->get_ne(1); i1++) {
                // TODO: prefetch row?
                auto * src0_row = src0_plane + i1 * src0->get_nb(1);
                auto * src1_row = src1_plane + i1 / r11 * src1->get_nb(1);
                auto * dst_row  = reinterpret_cast<float *>(dst_plane + i1 * out->get_nb(1));
                _RowFunc(reinterpret_cast<const _TySrc *>(src0_row), reinterpret_cast<const _TySrc *>(src1_row),
                         static_cast<size_t>(out->get_ne(0)), reinterpret_cast<_TyDst *>(dst_row));
            }
        }
    }

    return true;
}

constexpr const hexagon::compute_func_t kOpArray[] = {
    mul_mat_f32,                                                  // NPU_OP_MUL_MAT
    element_wise_op<float, float, vec_op_f32_f32<vadd_f32_f32>>,  // NPU_OP_ADD
    element_wise_op<float, float, vec_op_f32_f32<vsub_f32_f32>>,  // NPU_OP_SUB
    element_wise_op<float, float, vec_op_f32_f32<vmul_f32_f32>>,  // NPU_OP_MUL
};

static_assert(kOpArray[NPU_OP_MUL_MAT] == mul_mat_f32, "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert((sizeof(kOpArray) / sizeof(kOpArray[0])) == NPU_OP_COUNT);

}  // namespace

namespace hexagon {

compute_func_t get_compute_func(npu_device_tensor_op op) {
    return kOpArray[op];
}

}  // namespace hexagon
