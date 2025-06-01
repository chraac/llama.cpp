#include "vec_ops.hpp"

#include <HTP/core/intrinsics.h>

#include "util.hpp"

namespace hexagon {

float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector * src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * src0_vec_ptr_end = ((HVX_Vector *) src0) + count / kElementsPerVector;
    HVX_Vector * src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector   prev0            = *src0_vec_ptr++;
    HVX_Vector   prev1            = *src1_vec_ptr++;
    HVX_Vector   sum              = Q6_V_vzero();

    while (src0_vec_ptr_end - src0_vec_ptr > 1) {
        HVX_Vector curr0_lo = src0_vec_ptr[0];
        HVX_Vector curr0_hi = src0_vec_ptr[1];
        HVX_Vector curr1_lo = src1_vec_ptr[0];
        HVX_Vector curr1_hi = src1_vec_ptr[1];

        HVX_Vector l0 = Q6_V_valign_VVR(curr0_lo, prev0, (size_t) src0);
        HVX_Vector l1 = Q6_V_valign_VVR(curr1_lo, prev1, (size_t) src1);
        HVX_Vector h0 = Q6_V_valign_VVR(curr0_hi, curr0_lo, (size_t) src0);
        HVX_Vector h1 = Q6_V_valign_VVR(curr1_hi, curr1_lo, (size_t) src1);
        prev0         = curr0_hi;
        prev1         = curr1_hi;
        src0_vec_ptr += 2;
        src1_vec_ptr += 2;

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(l0, l1), sum);
        sum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(h0, h1), sum);
    }

    if (src0_vec_ptr_end - src0_vec_ptr > 0) {
        HVX_Vector curr0 = *src0_vec_ptr++;
        HVX_Vector curr1 = *src1_vec_ptr++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0            = curr0;
        prev1            = curr1;

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
    }

    if ((src0_vec_ptr_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool       iptr0_aligned = hexagon::is_addr_aligned(src0_vec_ptr);
        HVX_Vector curr0         = iptr0_aligned ? prev0 : *src0_vec_ptr;
        src0_vec_ptr             = iptr0_aligned ? src0_vec_ptr : src0_vec_ptr + 1;
        bool       iptr1_aligned = hexagon::is_addr_aligned(src1_vec_ptr);
        HVX_Vector curr1         = iptr1_aligned ? prev1 : *src1_vec_ptr;
        src1_vec_ptr             = iptr1_aligned ? src1_vec_ptr : src1_vec_ptr + 1;
        HVX_Vector s0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0                    = curr0;
        prev1                    = curr1;

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 = (leftover_bytes + hexagon::unaligned_bytes(src0_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src0_vec_ptr :
                               prev0;
        curr0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_V_valign_VVR(Q6_Vqf32_vmpy_VsfVsf(curr0, curr1), Q6_V_vzero(), leftover_bytes), sum);
    }

    return hexagon::vec_reduction_f32(sum);
}

// TODO: merge with vec_dot_product_f32_f32?
float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(npu_device_fp16_t);
    constexpr const size_t kFloatsPerVector   = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector * src0_vec_ptr     = ((HVX_Vector *) src0);
    HVX_Vector * src0_vec_ptr_end = ((HVX_Vector *) src0) + (count / kElementsPerVector);
    HVX_Vector * src1_vec_ptr     = ((HVX_Vector *) src1);
    HVX_Vector   prev0            = *src0_vec_ptr++;
    HVX_Vector   prev1            = *src1_vec_ptr++;
    HVX_Vector   sum_hi           = Q6_V_vzero();
    HVX_Vector   sum_lo           = Q6_V_vzero();

    while (src0_vec_ptr < src0_vec_ptr_end) {
        HVX_Vector curr0 = *src0_vec_ptr++;
        HVX_Vector curr1 = *src1_vec_ptr++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0            = curr0;
        prev1            = curr1;

        HVX_VectorPair result = Q6_Wqf32_vmpy_VhfVhf(s0, s1);
        sum_hi                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(result), sum_hi);
        sum_lo                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(result), sum_lo);
    }

    if ((src0_vec_ptr_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool       iptr0_aligned = hexagon::is_addr_aligned(src0_vec_ptr);
        HVX_Vector curr0         = iptr0_aligned ? prev0 : *src0_vec_ptr;
        src0_vec_ptr             = iptr0_aligned ? src0_vec_ptr : src0_vec_ptr + 1;
        bool       iptr1_aligned = hexagon::is_addr_aligned(src1_vec_ptr);
        HVX_Vector curr1         = iptr1_aligned ? prev1 : *src1_vec_ptr;
        src1_vec_ptr             = iptr1_aligned ? src1_vec_ptr : src1_vec_ptr + 1;
        HVX_Vector s0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        prev0                    = curr0;
        prev1                    = curr1;

        HVX_VectorPair result = Q6_Wqf32_vmpy_VhfVhf(s0, s1);
        sum_hi                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(result), sum_hi);
        sum_lo                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(result), sum_lo);
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(npu_device_fp16_t);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 = (leftover_bytes + hexagon::unaligned_bytes(src0_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src0_vec_ptr :
                               prev0;
        HVX_Vector curr1 = (leftover_bytes + hexagon::unaligned_bytes(src1_vec_ptr) > hexagon::kBytesPerVector) ?
                               *src1_vec_ptr :
                               prev1;

        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
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

    return hexagon::vec_reduction_f32(Q6_Vqf32_vadd_Vqf32Vqf32(sum_hi, sum_lo));
}

}  // namespace hexagon
