#pragma once

#include "util.hpp"

#include <dma_desc.h>
#include <dma_utils.h>
#include <qurt.h>

namespace hexagon::dma {

class dma_transfer {
  public:
    dma_transfer() {
        dma_desc_set_next(_dma_desc, NULL);
        dma_desc_set_dstate(_dma_desc, DESC_DSTATE_INCOMPLETE);
        dma_desc_set_desctype(_dma_desc, DMA_DESC_TYPE_1D);

        dma_desc_set_order(_dma_desc, 1);
        dma_desc_set_bypasssrc(_dma_desc, 1);  // for dram
        dma_desc_set_bypassdst(_dma_desc, 0);  // for vtcm
        dma_desc_set_length(_dma_desc, 0);
    }

    ~dma_transfer() { wait(); }

    bool submit(const uint8_t * src, uint8_t * dst, size_t size) {
        if (dma_desc_is_done(_dma_desc) == DMA_INCOMPLETE) {
            return false;
        }

        dma_desc_set_src(_dma_desc, reinterpret_cast<uint32_t>(src));
        dma_desc_set_dst(_dma_desc, reinterpret_cast<uint32_t>(dst));
        dma_desc_set_length(_dma_desc, size);

        _dma_desc_mutex.lock();
        void * buffs[1] = { _dma_desc };
        if (dma_desc_submit(buffs, 1) != DMA_SUCCESS) {
            _dma_desc_mutex.unlock();
            return false;
        }

        _dma_desc_mutex.unlock();
        return true;
    }

    void wait() { dma_wait_for_idle(); }

  private:
    uint8_t _dma_desc[DMA_DESC_SIZE_1D] __attribute__((aligned(16))) = {};

    static qurt_mutex _dma_desc_mutex;

    DISABLE_COPY_AND_MOVE(dma_transfer);
};

}  // namespace hexagon::dma
