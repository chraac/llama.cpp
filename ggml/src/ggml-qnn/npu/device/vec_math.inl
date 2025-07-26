#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

namespace hexagon::vec::math {

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

inline HVX_VectorPair hvx_vqf32_convert_vhf(HVX_Vector vxl) {
    return qhmath_hvx_vqf32_convert_vqf16(qhmath_hvx_vqf16_convert_vhf(vxl));
}

inline HVX_Vector_x2 hvx_vsf_convert_vhf(HVX_Vector vxl, HVX_Vector one) {
    HVX_VectorPair res = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vxl), one);
    return {
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(res)),
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(res)),
    };
}

}  // namespace hexagon::vec::math
