#include "dma_transfer.hpp"

#include <dma_desc.h>
#include <qurt.h>

#include <array>
#include <cstdlib>

namespace hexagon::dma {

dma_transfer::dma_transfer() {
    dma_desc_set_next(_dma_desc0, 0);
    dma_desc_set_dstate(_dma_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_desc0, DMA_DESC_TYPE_1D);

    dma_desc_set_order(_dma_desc0, 1);
    dma_desc_set_bypasssrc(_dma_desc0, 1);  // for dram
    dma_desc_set_bypassdst(_dma_desc0, 0);  // for vtcm
    dma_desc_set_length(_dma_desc0, 0);

    dma_desc_set_next(_dma_desc1, 0);
    dma_desc_set_dstate(_dma_desc1, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_desc1, DMA_DESC_TYPE_1D);

    dma_desc_set_order(_dma_desc1, 1);
    dma_desc_set_bypasssrc(_dma_desc1, 1);  // for dram
    dma_desc_set_bypassdst(_dma_desc1, 0);  // for vtcm
    dma_desc_set_length(_dma_desc1, 0);
}

dma_transfer::~dma_transfer() {
    wait();
}

bool dma_transfer::submit(const uint8_t * src, uint8_t * dst, size_t size) {
    if (!dma_transfer::is_desc_done(_dma_desc0)) {
        DEVICE_LOG_ERROR("Failed to initiate DMA transfer for one or more descriptors\n");
        return false;
    }

    dma_desc_set_next(_dma_desc0, 0);
    dma_desc_set_dstate(_dma_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_desc0, reinterpret_cast<uint32_t>(src));
    dma_desc_set_dst(_dma_desc0, reinterpret_cast<uint32_t>(dst));
    dma_desc_set_length(_dma_desc0, size);

    void * buffs[] = { _dma_desc0 };
    _dma_desc_mutex.lock();
    if (dma_desc_submit(buffs, std::size(buffs)) != DMA_SUCCESS) {
        _dma_desc_mutex.unlock();
        DEVICE_LOG_ERROR("Failed to submit DMA descriptor\n");
        return false;
    }

    _dma_desc_mutex.unlock();
    return true;
}

bool dma_transfer::submit(const uint8_t * src0, uint8_t * dst0, const uint8_t * src1, uint8_t * dst1, size_t size) {
    if (!dma_transfer::is_desc_done(_dma_desc0) || !dma_transfer::is_desc_done(_dma_desc1)) {
        DEVICE_LOG_ERROR("Failed to initiate DMA transfer for one or more descriptors\n");
        return false;
    }

    dma_desc_set_next(_dma_desc0, 0);
    dma_desc_set_dstate(_dma_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_desc0, reinterpret_cast<uint32_t>(src0));
    dma_desc_set_dst(_dma_desc0, reinterpret_cast<uint32_t>(dst0));
    dma_desc_set_length(_dma_desc0, size);

    dma_desc_set_next(_dma_desc1, 0);
    dma_desc_set_dstate(_dma_desc1, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_desc1, reinterpret_cast<uint32_t>(src1));
    dma_desc_set_dst(_dma_desc1, reinterpret_cast<uint32_t>(dst1));
    dma_desc_set_length(_dma_desc1, size);

    void * buffs[] = { _dma_desc0, _dma_desc1 };
    _dma_desc_mutex.lock();
    if (dma_desc_submit(buffs, std::size(buffs)) != DMA_SUCCESS) {
        _dma_desc_mutex.unlock();
        DEVICE_LOG_ERROR("Failed to submit DMA descriptor\n");
        return false;
    }

    _dma_desc_mutex.unlock();
    return true;
}

void dma_transfer::wait() {
    auto ret = dma_wait_for_idle();
    if (ret != DMA_SUCCESS) {
        DEVICE_LOG_ERROR("dma_transfer: failed to wait for DMA idle: %d\n", ret);
    }
}

bool dma_transfer::is_desc_done(uint8_t * desc) {
    return !dma_desc_get_src(desc) || dma_desc_is_done(desc) == DMA_COMPLETE;
}

qurt_mutex dma_transfer::_dma_desc_mutex = {};

}  // namespace hexagon::dma
