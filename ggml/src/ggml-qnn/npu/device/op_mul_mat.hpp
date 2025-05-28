#pragma once

#include <hexagon_types.h>

#include "op_types.hpp"
#include "tensor.hpp"

namespace hexagon {

inline void l2fetch(const void * p, uint32_t stride, uint32_t width, uint32_t height, uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

inline void l2fetch_row(const uint8_t * curr_row, size_t bytes) {
    // TODO: should we use small kL2FetchAheadVectors?
    int32_t l2fetch_vectors = Q6_R_min_RR(bytes / kBytesPerVector, kL2FetchAheadVectors);
    hexagon::l2fetch(curr_row, kBytesPerVector, kBytesPerVector, l2fetch_vectors, 0);
}

bool mul_mat_f32(tensor * out, compute_params * params);
bool is_mul_mat_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                          const npu_device_tensor_spec * srcs, size_t src_len);

}  // namespace hexagon
