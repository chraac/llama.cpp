#pragma once

#include <hexagon_types.h>

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
    return hexagon::get_flt0_from_fltv(Q6_Vsf_equals_Vqf32(vec_reduction_qf32(sums)));
}

float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count);

float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count);

}  // namespace hexagon
