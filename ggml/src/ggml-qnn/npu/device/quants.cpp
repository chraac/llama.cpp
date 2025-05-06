#include "quants.hpp"

#include <hexagon_types.h>

#include <array>

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

void dequantize_row_q4_K(const npu_device_block_q4_K * src, float * dst, size_t count, const float * f16_to_f32_table) {
    const auto nb = count / QUANT_K_BLOCK_SIZE;

    // TODO: use intrinsics
    for (size_t i = 0; i < nb; i++) {
        const uint8_t * q = src[i].qs;

        const float d   = f16_to_f32_table[src[i].d];
        const float min = f16_to_f32_table[src[i].dmin];

        int          is     = 0;
        uint8_t      sc     = 0;
        uint8_t      m      = 0;
        const auto * scales = src[i].scales;
        for (int j = 0; j < QUANT_K_BLOCK_SIZE; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
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

constexpr const hexagon::device_type_traits kDeviceTypeTraits[] = {
    { NPU_DATA_TYPE_F32,  "F32",  1,                  false, nullptr             },
    { NPU_DATA_TYPE_F16,  "F16",  1,                  false, nullptr             },
    { NPU_DATA_TYPE_Q8_0, "Q8_0", QUANT_BLOCK_SIZE,   true,  nullptr             },
    { NPU_DATA_TYPE_Q4_0, "Q4_0", QUANT_BLOCK_SIZE,   true,  nullptr             },
    { NPU_DATA_TYPE_Q4_K, "Q4_K", QUANT_K_BLOCK_SIZE, true,  dequantize_row_q4_K },
};

static_assert(std::size(kDeviceTypeTraits) == NPU_DATA_TYPE_COUNT,
              "kDeviceTypeTraits size mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_F32].type == NPU_DATA_TYPE_F32,
              "kDeviceTypeTraits F32 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_F16].type == NPU_DATA_TYPE_F16,
              "kDeviceTypeTraits F16 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_Q8_0].type == NPU_DATA_TYPE_Q8_0,
              "kDeviceTypeTraits Q8_0 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_Q4_0].type == NPU_DATA_TYPE_Q4_0,
              "kDeviceTypeTraits Q4_0 type mismatch with npu_device_tensor_data_type enum");
static_assert(kDeviceTypeTraits[NPU_DATA_TYPE_Q4_K].type == NPU_DATA_TYPE_Q4_K,
              "kDeviceTypeTraits Q4_K type mismatch with npu_device_tensor_data_type enum");

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

    return true;
}

const device_type_traits & get_type_traits(npu_device_tensor_data_type type) {
    return kDeviceTypeTraits[type];
}

}  // namespace hexagon
