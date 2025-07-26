#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

// TODO: move this macros to a common header
#define IEEE_VSF_EXPLEN   (8)
#define IEEE_VSF_EXPBIAS  (127)
#define IEEE_VSF_EXPMASK  (0xFF)
#define IEEE_VSF_MANTLEN  (23)
#define IEEE_VSF_MANTMASK (0x7FFFFF)
#define IEEE_VSF_MIMPMASK (0x800000)

#define IEEE_VHF_EXPLEN   (5)
#define IEEE_VHF_EXPBIAS  (15)
#define IEEE_VHF_EXPMASK  (0x1F)
#define IEEE_VHF_MANTLEN  (10)
#define IEEE_VHF_MANTMASK (0x3FF)
#define IEEE_VHF_MIMPMASK (0x400)

#define COEFF_EXP_5 0x39506967  // 0.000198757 = 1/(7!)
#define COEFF_EXP_4 0x3AB743CE  // 0.0013982   = 1/(6!)
#define COEFF_EXP_3 0x3C088908  // 0.00833345  = 1/(5!)
#define COEFF_EXP_2 0x3D2AA9C1  // 0.416658    = 1/(4!)
#define COEFF_EXP_1 0x3E2AAAAA  // 0.16666667  = 1/(3!)
#define COEFF_EXP_0 0x3F000000  // 0.5         = 1/(2!)
#define LOGN2       0x3F317218  // ln(2)   = 0.6931471805
#define LOG2E       0x3FB8AA3B  // log2(e) = 1/ln(2) = 1.4426950408

#define COEFF_EXP_5_HF 0x0A83   // 0.000198757 = 1/(7!)
#define COEFF_EXP_4_HF 0x15BA   // 0.0013982   = 1/(6!)
#define COEFF_EXP_3_HF 0x2044   // 0.00833345  = 1/(5!)
#define COEFF_EXP_2_HF 0x36AB   // 0.416658    = 1/(4!)
#define COEFF_EXP_1_HF 0x3155   // 0.16666667  = 1/(3!)
#define COEFF_EXP_0_HF 0x3800   // 0.5         = 1/(2!)
#define LOGN2_HF       0x398C   // ln(2)   = 0.693147
#define LOG2E_HF       0x3DC5   // log2(e) = 1/ln(2) = 1.4427

namespace hexagon::vec::math {

inline HVX_Vector qhmath_hvx_vsf_floor_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    // initialization (no changes)
    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, vin);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qneg) // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute floor
    mask_mant_v >>= expval_v;
    HVX_Vector neg_addin_v    = mask_impl_v >> expval_v;
    HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(vin, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, vin);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);               // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

//  truncate(x)
//  given a vector of float x,
//  return the vector of integers resulting from dropping all fractional bits
//  no checking performed for overflow - could be extended to return maxint
//
// truncate float to int
inline HVX_Vector qhmath_hvx_vw_truncate_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v = Q6_V_vzero();

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, vin);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_Vector rshift_v = IEEE_VSF_MANTLEN - expval_v;                // fractional bits - exp shift

    HVX_Vector mant_v = vin & mask_mant_v;                            // obtain mantissa
    HVX_Vector vout   = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);         // add implicit 1.0
    vout              = Q6_Vw_vasr_VwVw(vout, rshift_v);              // shift to obtain truncated integer
    vout              = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);  // expval<0 -> 0

    HVX_Vector neg_vout = -vout;
    vout                = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives
    return (vout);
}

// qhmath_hvx_vhf_floor_vhf(x)
//  given a vector of half float x,
//  return the vector of largest integer valued half float <= x
//
inline HVX_Vector qhmath_hvx_vhf_floor_vhf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector const_emask_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_Vh_vsplat_R(0xbc00);  // -1 IEEE vhf

    // initialization (no changes)
    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VhVh(const_zero_v, vin);

    HVX_Vector expval_v = Q6_Vh_vasr_VhR(vin, IEEE_VHF_MANTLEN);
    expval_v            = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v            = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VhVh(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVhVh(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVhVh(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qneg) // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute floor
    mask_mant_v               = Q6_Vh_vasr_VhVh(mask_mant_v, expval_v);
    HVX_Vector neg_addin_v    = Q6_Vh_vasr_VhVh(mask_impl_v, expval_v);
    HVX_Vector vout_neg_addin = Q6_Vh_vadd_VhVh(vin, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, vin);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VhVh(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);               // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

// truncate half float to short
inline HVX_Vector qhmath_hvx_vh_truncate_vhf(HVX_Vector vin) {
    HVX_Vector const_mnlen_v = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector mask_mant_v   = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v   = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_emask_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v  = Q6_V_vzero();
    HVX_Vector const_one_v   = Q6_Vh_vsplat_R(1);

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VhVh(const_zero_v, vin);

    HVX_Vector expval_v = Q6_Vh_vasr_VhVh(vin, const_mnlen_v);
    expval_v            = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v            = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);

    // fractional bits - exp shift
    HVX_Vector rshift_v = Q6_Vh_vsub_VhVh(const_mnlen_v, expval_v);

    HVX_Vector mant_v = vin & mask_mant_v;                            // obtain mantissa
    HVX_Vector vout   = Q6_Vh_vadd_VhVh(mant_v, mask_impl_v);         // add implicit 1.0
    vout              = Q6_Vh_vasr_VhVh(vout, rshift_v);              // shift to obtain truncated integer
    vout              = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);  // expval<0 -> 0

    // HVX_Vector neg_vout = -vout;
    HVX_Vector not_vout = Q6_V_vnot_V(vout);
    HVX_Vector neg_vout = Q6_Vh_vadd_VhVh(not_vout, const_one_v);
    vout                = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives
    return (vout);
}

/*
 * This function computes the exponent on all IEEE 32-bit float elements of an HVX_Vector
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_exp_vf(HVX_Vector sline) {
    HVX_Vector z_qf32_v;
    HVX_Vector x_v;
    HVX_Vector x_qf32_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector epsilon_v;
    HVX_Vector log2e = Q6_V_vsplat_R(LOG2E);
    HVX_Vector logn2 = Q6_V_vsplat_R(LOGN2);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // 1) clipping + uint input
    //        if (x > MAXLOG)
    //            return (MAXNUM);
    //        if (x < MINLOG)
    //            return (0.0);
    //
    // 2) exp(x) is approximated as follows:
    //   f = floor(x/ln(2)) = floor(x*log2(e))
    //   epsilon = x - f*ln(2)
    //   exp(x) = exp(epsilon+f*ln(2))
    //          = exp(epsilon)*exp(f*ln(2))
    //          = exp(epsilon)*2^f
    //   Since epsilon is close to zero, it can be approximated with its Taylor series:
    //            exp(x)~=1+x+x^2/2!+x^3/3!+...+x^n/n!+...
    //   Preserving the first eight elements, we get:
    //            exp(x)~=1+x+e0*x^2+e1*x^3+e2*x^4+e3*x^5+e4*x^6+e5*x^7
    //                   =1+x+(E0+(E1+(E2+(E3+(E4+E5*x)*x)*x)*x)*x)*x^2

    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, sline);
    epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = qhmath_hvx_vsf_floor_vsf(epsilon_v);
    k_v = qhmath_hvx_vw_truncate_vsf(f_v);

    x_qf32_v = Q6_Vqf32_vadd_VsfVsf(sline, zero_v);

    //    x = x - f_v * logn2;
    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
    x_qf32_v  = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
    //    normalize before every QFloat's vmpy
    x_qf32_v  = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);

    //    z = x * x;
    z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
    z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

    x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

    //    y = E4 + E5 * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_5);
    y_v     = Q6_Vqf32_vmpy_VsfVsf(E_const, x_v);
    E_const = Q6_V_vsplat_R(COEFF_EXP_4);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E3 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_3);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E2 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_2);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E1 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_1);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E0 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_0);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = x + y * z;
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = y + 1.0;
    E_const = Q6_V_vsplat_R(0x3f800000);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);

    //insert exponents
    //        y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent
    y_v = Q6_Vsf_equals_Vqf32(y_v);

    //    add k_v to the exponent of y_v
    HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

    y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, 24);

    y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    //    exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, 23);

    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

    return y_v;
}

/*
 * This function computes the exponent on all IEEE 16-bit float elements of an HVX_Vector
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_exp_vhf(HVX_Vector sline) {
    HVX_Vector z_qf16_v;
    HVX_Vector x_qf16_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector tmp_v;
    HVX_Vector log2e = Q6_Vh_vsplat_R(LOG2E_HF);
    HVX_Vector logn2 = Q6_Vh_vsplat_R(LOGN2_HF);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // 1) clipping + uint input
    //        if (x > MAXLOG)
    //            return (MAXNUM);
    //        if (x < MINLOG)
    //            return (0.0);

    // 2) round to int
    //    k = (int) (x * log2e);
    //    f = (float) k;
    //    k = Q6_R_convert_sf2w_R(log2e * x); //f = floorf( log2e * x + 0.5);
    //    f = Q6_R_convert_w2sf_R(k);         //k = (int)f;

    tmp_v            = Q6_Vqf16_vmpy_VhfVhf(log2e, sline);
    //    float16's 0.5 is 0x3800
    HVX_Vector cp5_v = Q6_Vh_vsplat_R(0x3800);
    tmp_v            = Q6_Vqf16_vadd_Vqf16Vhf(tmp_v, cp5_v);
    tmp_v            = Q6_Vhf_equals_Vqf16(tmp_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = qhmath_hvx_vhf_floor_vhf(tmp_v);
    k_v = qhmath_hvx_vh_truncate_vhf(f_v);

    x_qf16_v = Q6_Vqf16_vadd_VhfVhf(sline, zero_v);

    //    x = x - f * logn2;
    tmp_v    = Q6_Vqf16_vmpy_VhfVhf(f_v, logn2);
    x_qf16_v = Q6_Vqf16_vsub_Vqf16Vqf16(x_qf16_v, tmp_v);

    //    normalize before every QFloat's vmpy
    x_qf16_v = Q6_Vqf16_vadd_Vqf16Vhf(x_qf16_v, zero_v);

    //    z = x * x;
    z_qf16_v = Q6_Vqf16_vmpy_Vqf16Vqf16(x_qf16_v, x_qf16_v);
    z_qf16_v = Q6_Vqf16_vadd_Vqf16Vhf(z_qf16_v, zero_v);

    //    y = E4 + E5 * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_5_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vhf(x_qf16_v, E_const);
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_4_HF);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E3 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_3_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E2 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_2_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E1 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_1_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E0 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_0_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = x + y * z;
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, z_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = y + 1.0;
    E_const = Q6_Vh_vsplat_R(0x3C00);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);

    // insert exponents
    //    y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent
    y_v = Q6_Vhf_equals_Vqf16(y_v);

    // add k_v to the exponent of y_v
    // shift away sign bit
    HVX_Vector y_v_exponent = Q6_Vh_vasl_VhR(y_v, 1);

    // shift back by sign bit + 10-bit mantissa
    y_v_exponent = Q6_Vuh_vlsr_VuhR(y_v_exponent, 11);

    y_v_exponent = Q6_Vh_vadd_VhVh(k_v, y_v_exponent);

    // exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VhVh(zero_v, y_v_exponent);

    // max IEEE hf exponent; if overflow detected, result is set to infinity
    HVX_Vector     exp_max_v              = Q6_Vh_vsplat_R(0x1e);
    // INF in 16-bit float is 0x7C00
    HVX_Vector     inf_v                  = Q6_Vh_vsplat_R(0x7C00);
    HVX_VectorPred qy_v_overflow_exponent = Q6_Q_vcmp_gt_VhVh(y_v_exponent, exp_max_v);

    // update exponent
    y_v = Q6_Vh_vaslacc_VhVhR(y_v, k_v, 10);

    // clip to min/max values
    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
    y_v = Q6_V_vmux_QVV(qy_v_overflow_exponent, inf_v, y_v);

    return y_v;
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
