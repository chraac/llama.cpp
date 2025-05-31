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

inline HVX_UVector Q6_V_vmemu_R(const void * unaligned_ptr) {
    return *reinterpret_cast<const HVX_UVector *>(unaligned_ptr);
}

inline HVX_Vector Q6_V_vmem_R(const void * aligned_ptr) {
    return *reinterpret_cast<const HVX_Vector *>(aligned_ptr);
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
    HVX_Vector   prev           = *src_vec_ptr++;
    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);

    HVX_Vector scale_vec = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));

    while (src_vec_ptr < src_vec_end) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec));
        dst_vec_ptr++;
        prev = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       src_ptr_aligned = hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr            = src_ptr_aligned ? prev : *src_vec_ptr;
        src_vec_ptr                = src_ptr_aligned ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector s0              = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]             = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec));
        dst_vec_ptr++;
        prev = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(curr, scale_vec)));
    }
}

inline void vec_mad_f32(const float * src, float scale, float * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(float);

    HVX_Vector *  src_vec_ptr    = ((HVX_Vector *) src);
    HVX_Vector *  src_vec_end    = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_UVector * dst_vec_ptr    = ((HVX_UVector *) dst);  // TODO: opt the unaligned case?
    HVX_Vector    prev           = *src_vec_ptr++;
    const size_t  leftover       = count % kElementsPerVector;
    const size_t  leftover_bytes = leftover * sizeof(float);

    HVX_Vector scale_vec = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));
    while (src_vec_ptr < src_vec_end) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector d0   = dst_vec_ptr[0];  // TODO: opt the unaligned case?
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        s0              = Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec);
        s0              = Q6_Vqf32_vadd_Vqf32Vsf(s0, d0);
        dst_vec_ptr[0]  = Q6_Vsf_equals_Vqf32(s0);
        dst_vec_ptr++;
        prev = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       src_ptr_aligned = hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr            = src_ptr_aligned ? prev : *src_vec_ptr;
        src_vec_ptr                = src_ptr_aligned ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector d0              = dst_vec_ptr[0];
        HVX_Vector s0              = Q6_V_valign_VVR(curr, prev, (size_t) src);
        s0                         = Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec);
        s0                         = Q6_Vqf32_vadd_Vqf32Vsf(s0, d0);
        dst_vec_ptr[0]             = Q6_Vsf_equals_Vqf32(s0);
        dst_vec_ptr++;
        prev = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector d0 = dst_vec_ptr[0];  // TODO: opt the unaligned case?
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        curr = Q6_Vqf32_vmpy_VsfVsf(curr, scale_vec);
        curr = Q6_Vqf32_vadd_Vqf32Vsf(curr, d0);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, Q6_Vsf_equals_Vqf32(curr));
    }
}

/*
 * This function converts a vector of IEEE float elements to a vector of qf32 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vqf32_convert_vsf(HVX_Vector vin) {
    return Q6_Vqf32_vadd_VsfVsf(vin, Q6_V_vzero());
}

/*
 * This function converts a vector of IEEE half float elements to a vector of qf16 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vqf16_convert_vhf(HVX_Vector vin) {
    return Q6_Vqf16_vadd_VhfVhf(vin, Q6_V_vzero());
}

/*
 * This function converts a pair of vectors of qf32 elements to a vector of IEEE half float elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vhf_convert_vqf32(HVX_VectorPair vin_vp) {
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(vin_vp));
}

/*
 * This function converts a vector of qf16 elements to a pair of vectors of qf32 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_VectorPair qhmath_hvx_vqf32_convert_vqf16(HVX_Vector vxl) {
    HVX_VectorPair vxw_vp, exponent_vp;
    HVX_Vector     mantissa_mask = Q6_Vh_vsplat_R(0xffe0);
    HVX_Vector     exp_mask      = Q6_Vh_vsplat_R(0x1f);
    HVX_Vector     exp_offset    = Q6_Vh_vsplat_R(0x70);
    HVX_Vector     mant32_shift  = Q6_Vh_vsplat_R(0x10);
    HVX_Vector     reql, reqh, vxl_w, vxh_w, mantissa;
    HVX_Vector     el_exponent, eh_exponent;

    el_exponent = Q6_V_vand_VV(exp_mask, vxl);
    // Obtain the mantissa part: bits (5-15)
    mantissa    = Q6_V_vand_VV(mantissa_mask, vxl);
    // Convert qf16 biassed exponent to qf32 biased exponent
    // new exp = exp + ( 127 (qf32 bias) -15(qf16 biass) ) = 112
    el_exponent = Q6_Vh_vadd_VhVh(exp_offset, el_exponent);

    vxw_vp = Q6_Ww_vunpack_Vh(mantissa);
    vxl_w  = Q6_V_lo_W(vxw_vp);
    vxh_w  = Q6_V_hi_W(vxw_vp);

    exponent_vp = Q6_Ww_vunpack_Vh(el_exponent);
    el_exponent = Q6_V_lo_W(exponent_vp);
    eh_exponent = Q6_V_hi_W(exponent_vp);
    // Convert q16 mantiss to q32 mantissa
    reql        = Q6_Vw_vasl_VwVw(vxl_w, mant32_shift);
    reqh        = Q6_Vw_vasl_VwVw(vxh_w, mant32_shift);
    // Add the exponent
    vxl_w       = Q6_Vw_vadd_VwVw(reql, el_exponent);
    vxh_w       = Q6_Vw_vadd_VwVw(reqh, eh_exponent);

    return Q6_W_vcombine_VV(vxh_w, vxl_w);
}

inline HVX_Vector hvx_vec_scale_f16_qf32(HVX_Vector src, HVX_Vector scale_vec) {
    HVX_VectorPair src_pair = qhmath_hvx_vqf32_convert_vqf16(qhmath_hvx_vqf16_convert_vhf(src));
    HVX_Vector     lo       = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(src_pair), scale_vec);
    HVX_Vector     hi       = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(src_pair), scale_vec);
    src_pair                = Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(lo), Q6_Vsf_equals_Vqf32(hi));
    return qhmath_hvx_vhf_convert_vqf32(src_pair);  // TODO: can we avoid the vdeal?
}

inline HVX_Vector hvx_vec_mad_f16_qf32(HVX_Vector src, HVX_Vector dst, HVX_Vector scale_vec) {
    HVX_VectorPair src_pair = qhmath_hvx_vqf32_convert_vqf16(qhmath_hvx_vqf16_convert_vhf(src));
    HVX_Vector     lo       = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(src_pair), scale_vec);
    HVX_Vector     hi       = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(src_pair), scale_vec);
    src_pair                = Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(lo), Q6_Vsf_equals_Vqf32(hi));
    lo                      = qhmath_hvx_vhf_convert_vqf32(src_pair);  // TODO: can we avoid the vdeal?
    lo                      = Q6_Vqf16_vadd_Vqf16Vhf(lo, dst);
    return Q6_Vhf_equals_Vqf16(lo);
}

inline void vec_scale_f16(const npu_device_fp16_t * src, float scale, npu_device_fp16_t * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(npu_device_fp16_t);

    HVX_Vector *  src_vec_ptr    = ((HVX_Vector *) src);
    HVX_Vector *  src_vec_end    = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_UVector * dst_vec_ptr    = ((HVX_UVector *) dst);  // TODO: opt the unaligned case?
    HVX_Vector    prev           = *src_vec_ptr++;
    const size_t  leftover       = count % kElementsPerVector;
    const size_t  leftover_bytes = leftover * sizeof(float);

    HVX_Vector scale_vec = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));
    scale_vec            = qhmath_hvx_vqf32_convert_vsf(scale_vec);

    while (src_vec_ptr < src_vec_end) {
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]  = hvx_vec_scale_f16_qf32(s0, scale_vec);
        dst_vec_ptr++;
        prev = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        bool       src_ptr_aligned = hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr            = src_ptr_aligned ? prev : *src_vec_ptr;
        src_vec_ptr                = src_ptr_aligned ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector s0              = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]             = hvx_vec_scale_f16_qf32(s0, scale_vec);
        dst_vec_ptr++;
        prev = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, hvx_vec_scale_f16_qf32(curr, scale_vec));
    }
}

inline void vec_mad_f16(const npu_device_fp16_t * src, float scale, npu_device_fp16_t * dst, size_t count) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(npu_device_fp16_t);

    HVX_Vector *  src_vec_ptr    = ((HVX_Vector *) src);
    HVX_Vector *  src_vec_end    = ((HVX_Vector *) src) + (count / kElementsPerVector);
    HVX_UVector * dst_vec_ptr    = ((HVX_UVector *) dst);  // TODO: opt the unaligned case?
    HVX_Vector    prev           = *src_vec_ptr++;
    const size_t  leftover       = count % kElementsPerVector;
    const size_t  leftover_bytes = leftover * sizeof(float);

    HVX_Vector scale_vec = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(scale));
    scale_vec            = qhmath_hvx_vqf32_convert_vsf(scale_vec);

    while (src_vec_ptr < src_vec_end) {
        HVX_Vector d0   = dst_vec_ptr[0];  // TODO: opt the unaligned case?
        HVX_Vector curr = *src_vec_ptr++;
        HVX_Vector s0   = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]  = hvx_vec_mad_f16_qf32(s0, d0, scale_vec);
        dst_vec_ptr++;
        prev = curr;
    }

    if ((src_vec_end - ((HVX_Vector *) src)) > 0) {
        // handle the last vector
        HVX_Vector d0              = dst_vec_ptr[0];  // TODO: opt the unaligned case?
        bool       src_ptr_aligned = hexagon::is_addr_aligned(src_vec_ptr);
        HVX_Vector curr            = src_ptr_aligned ? prev : *src_vec_ptr;
        src_vec_ptr                = src_ptr_aligned ? src_vec_ptr : src_vec_ptr + 1;
        HVX_Vector s0              = Q6_V_valign_VVR(curr, prev, (size_t) src);
        dst_vec_ptr[0]             = hvx_vec_mad_f16_qf32(s0, d0, scale_vec);
        dst_vec_ptr++;
        prev = curr;
    }

    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector d0 = dst_vec_ptr[0];  // TODO: opt the unaligned case?
        HVX_Vector curr =
            (leftover_bytes + hexagon::unaligned_bytes(src_vec_ptr) > hexagon::kBytesPerVector) ? *src_vec_ptr : prev;
        curr = Q6_V_valign_VVR(curr, prev, (size_t) src);
        q6op_vstu_variable_ARV(dst_vec_ptr, leftover_bytes, hvx_vec_mad_f16_qf32(curr, d0, scale_vec));
    }
}

float vec_dot_product_f32_f32(const float * src0, const float * src1, size_t count);

float vec_dot_product_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count);

}  // namespace hexagon
