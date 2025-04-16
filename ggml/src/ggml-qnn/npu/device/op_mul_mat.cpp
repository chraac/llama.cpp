#include "op_mul_mat.hpp"

#include <HTP/core/intrinsics.h>

namespace {

inline float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count) {
    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / hexagon::kFloatsPerVector);
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
        HVX_Vector curr0 = hexagon::is_addr_aligned(iptr0) ? prev0 : *iptr0++;
        HVX_Vector curr1 = hexagon::is_addr_aligned(iptr1) ? prev1 : *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        sum              = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
        prev0            = curr0;
        prev1            = curr1;
    }

    const size_t leftover       = count % hexagon::kFloatsPerVector;
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
    for (size_t i = hexagon::kFloatsPerVector / 2; i > 0; i /= 2) {
        sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vror_VR(sum, i * sizeof(float)));
    }

    float result;
    q6op_vstu_variable_ARV(&result, sizeof(float), Q6_Vsf_equals_Vqf32(sum));
    return result;
}

}  // namespace

namespace hexagon {

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

bool is_mul_mat_supported(const npu_device_ne_type src0, const npu_device_ne_type src1, const npu_device_ne_type dst,
                          npu_device_tensor_op op) {
    if (op != NPU_OP_MUL_MAT) {
        DEVICE_LOG_DEBUG("op is not NPU_OP_MUL_MAT: %d\n", op);
        return false;
    }

    if (src0[0] != src1[0] || src0[1] != dst[0]) {
        DEVICE_LOG_DEBUG("src0 and src1 cannot multiply: %ldx%ld vs %ldx%ld\n", (long) src0[0], (long) src0[1],
                         (long) src1[0], (long) src1[1]);
        return false;
    }

    if (src1[1] != dst[1] || src1[2] != dst[2] || src1[3] != dst[3]) {
        DEVICE_LOG_DEBUG("src1 and dst dimensions not match: %ldx%ld vs %ldx%ld\n", (long) src1[2], (long) src1[3],
                         (long) dst[2], (long) dst[3]);
        return false;
    }

    if (src1[2] % src0[2] || src1[3] % src0[3]) {
        DEVICE_LOG_DEBUG("src0 cannot broadcast to src1: %ldx%ld vs %ldx%ld\n", (long) src0[2], (long) src0[3],
                         (long) src1[2], (long) src1[3]);
        return false;
    }

    return true;
}

}  // namespace hexagon
