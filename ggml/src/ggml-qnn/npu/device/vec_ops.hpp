#pragma once

#include <hexagon_types.h>
#include <HTP/core/intrinsics.h>

#include <cstdint>

#include "hexagon_npu.h"

namespace hexagon {

constexpr const size_t kBytesPerVector = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kAlignMask      = kBytesPerVector - 1;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(void * addr) {
    return unaligned_bytes(addr) == 0;
}

inline float get_flt0_from_fltv(HVX_Vector vect) {
    static_assert(sizeof(vect[0]) == sizeof(float), "vect[0] should be a float");
    int32_t i = vect[0];
    return reinterpret_cast<float &>(i);
}

inline HVX_Vector vec_reduction_qf32(HVX_Vector sums) {
    constexpr const size_t kFloatsPerVector = hexagon::kBytesPerVector / sizeof(float);
    static_assert(kFloatsPerVector == 32 || kFloatsPerVector == 16, "kFloatsPerVector should be 16 or 32");

    // TODO: do we have a better way to do the reduction?
    switch (kFloatsPerVector) {
        default:
        case 32:
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 16 * sizeof(float)));
            // fallthrough
        case 16:
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 8 * sizeof(float)));
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 4 * sizeof(float)));
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, 2 * sizeof(float)));
            sums = Q6_Vqf32_vadd_Vqf32Vqf32(sums, Q6_V_vror_VR(sums, sizeof(float)));
            break;
    }

    return sums;
}

inline float vec_reduction_f32(HVX_Vector sums) {
    return get_flt0_from_fltv(Q6_Vsf_equals_Vqf32(vec_reduction_qf32(sums)));
}

inline void vec_scale_f32(const float * src, float scale, float * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector * src_vec_ptr    = ((HVX_Vector *) src);
    HVX_Vector * src_vec_end    = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_Vector * dst_vec_ptr    = ((HVX_Vector *) dst);  // framework will ensure the dst is aligned
    HVX_Vector   scale_vec      = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));
    HVX_Vector   prev           = *src_vec_ptr++;
    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);

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

float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count);

float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count);

}  // namespace hexagon
