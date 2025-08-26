#pragma once

#include "util.hpp"

#include <dma_utils.h>

namespace hexagon::dma {

class dma_transfer {
  public:
    dma_transfer();

    ~dma_transfer() { wait(); }

    bool submit(const uint8_t * src, uint8_t * dst, size_t size);

    bool submit(const uint8_t * src0, uint8_t * dst0, const uint8_t * src1, uint8_t * dst1, size_t size);

    void wait();

  private:
    uint8_t _dma_desc0[DMA_DESC_SIZE_1D] __attribute__((aligned(16))) = {};
    uint8_t _dma_desc1[DMA_DESC_SIZE_1D] __attribute__((aligned(16))) = {};

    static qurt_mutex _dma_desc_mutex;

    DISABLE_COPY_AND_MOVE(dma_transfer);
};

}  // namespace hexagon::dma
