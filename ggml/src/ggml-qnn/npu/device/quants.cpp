#include "quants.hpp"

#include <hexagon_types.h>

namespace {

inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16
inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float    as_value;
    } fp32;

    fp32.as_bits = w;
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
    union {
        float    as_value;
        uint32_t as_bits;
    } fp32;

    fp32.as_value = f;
    return fp32.as_bits;
}

inline float get_fp32_from_fp16(npu_device_fp16_t h) {
    const uint32_t w     = (uint32_t) h << 16;
    const uint32_t sign  = w & uint32_t(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset       = uint32_t(0xE0) << 23;
    const float    exp_scale        = fp32_from_bits(uint32_t(0x80000000));
    const float    normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask         = uint32_t(126) << 23;
    const float    magic_bias         = 0.5f;
    const float    denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = uint32_t(1) << 27;
    const uint32_t result =
        sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

}  // namespace

namespace hexagon {

bool init_f16_f32_table(float * table, size_t count) {
    constexpr const size_t kTableSize = (1U << 16);
    if (count < kTableSize) {
        return false;
    }

    for (size_t i = 0; i < kTableSize; ++i) {
        table[i] = get_fp32_from_fp16(static_cast<npu_device_fp16_t>(i));
    }

    return true;
}

void dequantize_row_q4_K(const npu_device_block_q4_K * src, float * dst, size_t count, const float * f16_to_f32_table) {
    const auto nb = count / QUANT_K_BLOCK_SIZE;

    // TODO: refactor this to use the intrinsics
    for (size_t i = 0; i < nb; i++) {
        const uint8_t * q = src[i].qs;

        const float d   = f16_to_f32_table[src[i].d];
        const float min = f16_to_f32_table[src[i].dmin];

        int     is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE; j += 64) {
            get_scale_min_k4(is + 0, src[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, src[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *dst++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *dst++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

}  // namespace hexagon
