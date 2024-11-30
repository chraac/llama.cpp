#pragma once

#include <cstdint>
#include <memory>

#include "logger.hpp"
#include "qnn-lib.hpp"

namespace qnn {

class qnn_buffer_interface {
public:
    virtual ~qnn_buffer_interface() = default;

    virtual bool is_valid() const = 0;
    virtual uint8_t *get_buffer() = 0;
    virtual size_t get_size() const = 0;
    virtual Qnn_MemHandle_t get_mem_handle() const = 0;
};

using qnn_buffer_ptr = std::shared_ptr<qnn_buffer_interface>;

class qnn_rpc_buffer : public qnn_buffer_interface {
public:
    qnn_rpc_buffer(std::shared_ptr<qnn_instance> qnn_instance, const size_t size, const uint32_t rank,
                   uint32_t *dimensions, Qnn_DataType_t data_type)
        : _size(size), _qnn_instance(qnn_instance) {

        _qnn_rpc_buffer = static_cast<uint8_t *>(qnn_instance->alloc_rpcmem(size, alignof(uint8_t *)));
        _qnn_rpc_mem_handle = qnn_instance->register_rpcmem(_qnn_rpc_buffer, rank, dimensions, data_type);
        if (!_qnn_rpc_buffer || !_qnn_rpc_mem_handle) {
            QNN_LOG_WARN("register rpc mem failure");
            // let the destructor free the buffer
            return;
        }

        QNN_LOG_DEBUG("alloc rpcmem(%p) successfully, size %d", _qnn_rpc_buffer, (int)size);
    }
    ~qnn_rpc_buffer() {
        if (_qnn_instance) {
            if (_qnn_rpc_mem_handle) {
                _qnn_instance->unregister_rpcmem(_qnn_rpc_mem_handle);
            }

            if (_qnn_rpc_buffer) {
                _qnn_instance->free_rpcmem(_qnn_rpc_buffer);
            }
        }
    }

    bool is_valid() const override { return _qnn_rpc_buffer && _qnn_rpc_mem_handle; }

    uint8_t *get_buffer() override { return _qnn_rpc_buffer; }
    size_t get_size() const override { return _size; }
    Qnn_MemHandle_t get_mem_handle() const override { return _qnn_rpc_mem_handle; }

private:
    size_t _size = 0;
    uint8_t *_qnn_rpc_buffer = nullptr;
    Qnn_MemHandle_t _qnn_rpc_mem_handle = nullptr;
    std::shared_ptr<qnn_instance> _qnn_instance;

    DISABLE_COPY(qnn_rpc_buffer);
    DISABLE_MOVE(qnn_rpc_buffer);
};

class qnn_mem_buffer : public qnn_buffer_interface {
public:
    qnn_mem_buffer(size_t size) {
        _buffer = reinterpret_cast<uint8_t *>(qnn::page_align_alloc(size));

        if (!_buffer) {
            QNN_LOG_WARN("failed to allocate %.2f MiB", float(size / (1 << 20)));
            return;
        }

        _size = size;
    }

    ~qnn_mem_buffer() {
        // the free will do nothing if the _buffer is nullptr
        qnn::align_free(_buffer);
    }

    bool is_valid() const override { return _buffer != nullptr; }

    uint8_t *get_buffer() override { return _buffer; }
    size_t get_size() const override { return _size; }
    Qnn_MemHandle_t get_mem_handle() const override { return nullptr; }

private:
    size_t _size = 0;
    uint8_t *_buffer = nullptr;

    DISABLE_COPY(qnn_mem_buffer);
    DISABLE_MOVE(qnn_mem_buffer);
};

} // namespace qnn
