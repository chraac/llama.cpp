#pragma once

#include <hexagon_types.h>

#include <cstdint>

#include "tensor.hpp"

namespace hexagon {

constexpr const size_t kBytesPerVector      = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kAlignMask           = kBytesPerVector - 1;
constexpr const size_t kL2CacheSize         = 8 * 1024;            // // 8KB L2 cache
constexpr const size_t kL2FetchAheadVectors = kL2CacheSize / kBytesPerVector;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(void * addr) {
    return unaligned_bytes(addr) == 0;
}

inline void l2fetch(const void * p, uint32_t stride, uint32_t width, uint32_t height, uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

bool mul_mat_f32(tensor * out, size_t tidx, size_t tcnt);
bool is_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
