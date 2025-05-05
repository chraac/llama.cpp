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

inline void test_dequantize_row_q4_K(const float * f16_to_f32_table) {
    npu_device_block_q4_K src                     = {};
    float                 dst[QUANT_K_BLOCK_SIZE] = {};

    union {
        __fp16 f16;
        npu_device_fp16_t u16;
    } f16;

    f16.f16 = 0.5f;

    src.dmin = 0;
    src.d    = f16.u16;
    for (int i = 0; i < (QUANT_K_BLOCK_SIZE / 2); ++i) {
        src.qs[i] = 0x11;
    }

    for (int i = 0; i < QUANT_K_SCALE_SIZE; ++i) {
        src.scales[i] = 0xFF;
    }

    hexagon::dequantize_row_q4_K(&src, dst, QUANT_K_BLOCK_SIZE, f16_to_f32_table);

    DEVICE_LOG_DEBUG("dequantize_row_q4_K, {\n");
    for (int i = 0; i < QUANT_K_BLOCK_SIZE; i += 8) {
        DEVICE_LOG_DEBUG("    %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,\n", dst[i], dst[i + 1], dst[i + 2],
                         dst[i + 3], dst[i + 4], dst[i + 5], dst[i + 6], dst[i + 7]);
    }

    DEVICE_LOG_DEBUG("}\n");
}

}  // namespace

namespace hexagon {

bool init_f16_f32_table(float * table, size_t count) {
    constexpr const size_t kTableSize = (1U << 16);
    if (count < kTableSize) {
        return false;
    }

    union {
        __fp16 f16;
        npu_device_fp16_t u16;
    } f16;

    for (size_t i = 0; i < count; ++i) {
        f16.u16  = static_cast<npu_device_fp16_t>(i);
        table[i] = f16.f16;
    }

    // TODO: remove this test
    test_dequantize_row_q4_K(table);
    return true;
}

void dequantize_row_q4_K(const npu_device_block_q4_K * src, float * dst, size_t count, const float * f16_to_f32_table) {
    const auto nb = count / QUANT_K_BLOCK_SIZE;

    // TODO: use intrinsics
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
                dst[0]  = d1 * (q[l] & 0xF) - m1;
                dst[32] = d2 * ((q[l] >> 4) & 0xF) - m2;
                dst++;
            }
            dst += 32;
            q += 32;
            is += 2;
        }
    }
}

}  // namespace hexagon
