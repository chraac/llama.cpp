

#include <hexagon_types.h>

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

inline float vec_dot_product_f32(const float * src0, const float * src1, size_t count) {
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
        // see also: https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
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

    // TODO: add the 3d and 4d matrix support here
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return false;
    }

    if (src0->get_ne(0) != src1->get_ne(0)) {
        return false;
    }

    if (src0->get_ne(1) != out->get_ne(0)) {
        return false;
    }

    if (src1->get_ne(1) != out->get_ne(1)) {
        return false;
    }

    const auto   first_dim     = src0->get_ne(0);
    const auto   out_first_dim = out->get_ne(0);
    const auto * src0_ptr      = (float *) src0->get_data();
    const auto * src1_ptr      = (float *) src1->get_data();
    auto *       out_ptr       = (float *) out->get_data();
    for (int64_t i = 0; i < out->get_ne(1); i++) {
        // TODO: prefetch?
        auto * src1_row = src1_ptr + i * first_dim;
        auto * out_row  = out_ptr + i * out_first_dim;
        for (int64_t j = 0; j < out_first_dim; j++) {
            *out_row++ = vec_dot_product_f32(src0_ptr + j * first_dim, src1_row, (size_t) first_dim);
        }
    }

    return true;
}

constexpr const compute_func_t kOpArray[] = {
    mul_mat_f32,  // NPU_OP_MUL_MAT
};

static_assert((sizeof(kOpArray) / sizeof(kOpArray[0])) == NPU_OP_COUNT)

}  // namespace

namespace hexagon {

compute_func_t get_compute_func(npu_op op) {
    return kOpArray[op];
}

}  // namespace hexagon
