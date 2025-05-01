#pragma once

#include <cstdint>
#include <memory>

#include "hexagon_npu.h"
#include "tensor.hpp"
#include "vtcm_mem.hpp"

namespace hexagon {

struct compute_params {
    const size_t                       tidx;
    const size_t                       tcnt;
    const float *                      f16_to_f32_table;
    std::unique_ptr<hexagon::vtcm_mem> cache;

    uint8_t * get_cache(size_t size) {
        if (!cache || cache->get_size() < size) {
            cache = std::make_unique<hexagon::vtcm_mem>(size, false);
        }

        return cache->get_mem();
    }
};

typedef bool (*compute_func_type)(tensor * dst, compute_params * params);
typedef bool (*op_is_supported_func_type)(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
