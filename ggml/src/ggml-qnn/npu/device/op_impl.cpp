

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

    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    const auto   r2       = src1->get_ne(2) / src0->get_ne(2);
    const auto   r3       = src1->get_ne(3) / src0->get_ne(3);
    const auto * src0_ptr = reinterpret_cast<const uint8_t *>(src0->get_data());
    const auto * src1_ptr = reinterpret_cast<const uint8_t *>(src1->get_data());
    auto *       out_ptr  = reinterpret_cast<uint8_t *>(out->get_data());
    for (int64_t i3 = 0; i3 < out->get_ne(3); i3++) {
        const auto * src0_box = src0_ptr + i3 / r3 * src0->get_nb(3);
        const auto * src1_box = src1_ptr + i3 * src1->get_nb(3);
        auto *       out_box  = out_ptr + i3 * out->get_nb(3);
        for (int64_t i2 = 0; i2 < out->get_ne(2); i2++) {
            const auto * src0_plane = src0_box + i2 / r2 * src0->get_nb(2);
            const auto * src1_plane = src1_box + i2 * src1->get_nb(2);
            auto *       out_plane  = out_box + i2 * out->get_nb(2);
            for (int64_t i1 = 0; i1 < out->get_ne(1); i1++) {
                // TODO: prefetch row?
                auto * src1_row = src1_plane + i1 * src1->get_nb(1);
                auto * out_row  = reinterpret_cast<float *>(out_plane + i1 * out->get_nb(1));
                for (int64_t i0 = 0; i0 < out->get_ne(0); i0++) {
                    auto * src0_row = src0_plane + i0 * src0->get_nb(1);
                    // TODO: figure out how to handle a entire row
                    *out_row++ =
                        vec_dot_product_f32_f32(reinterpret_cast<const float *>(src0_row),
                                                reinterpret_cast<const float *>(src1_row), (size_t) src0->get_ne(0));
                }
            }
        }
    }

    out->flush();  // TODO: optimize this
    return true;
}

constexpr const hexagon::compute_func_t kOpArray[] = {
    mul_mat_f32,  // NPU_OP_MUL_MAT
};

static_assert(kOpArray[NPU_OP_MUL_MAT] == mul_mat_f32, "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert((sizeof(kOpArray) / sizeof(kOpArray[0])) == NPU_OP_COUNT);

}  // namespace

namespace hexagon {

compute_func_t get_compute_func(npu_device_tensor_op op) {
    return kOpArray[op];
}

}  // namespace hexagon
