
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "ggml-qnn.h"

#include "buffer.hpp"
#include "logger.hpp"
#include "qnn-lib.hpp"
#include "utils.hpp"

namespace qnn {

static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS should be 4");
using ggml_qnn_dimension_array_t = int64_t[GGML_MAX_DIMS];

class ggml_qnn_tensor {
public:
    typedef enum _tensor_type { INPUT, OUTPUT, INTERMEDIATE, PARAMETER } tensor_type_t;

    explicit ggml_qnn_tensor(tensor_type_t tensor_type, const std::string &name,
                             const ggml_qnn_dimension_array_t &dimensions, ggml_type data_type, int rank,
                             QNNBackend device, Qnn_GraphHandle_t graph_handle,
                             std::shared_ptr<qnn_instance> qnn_instance) :
        _tensor_name(name), _device(device), _qnn_instance(qnn_instance), _graph_handle(graph_handle) {
        QNN_TENSOR_SET_NAME(_qnn_tensor, _tensor_name.c_str());
        QNN_TENSOR_SET_DIMENSIONS(_qnn_tensor, _dimensions.data());
        QNN_TENSOR_SET_DATA_FORMAT(_qnn_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
        update_params_from_ggml_tensor(tensor_type, dimensions, data_type, rank);
        QNN_LOG_DEBUG("create tensor %s, type: %d, device: %d", _tensor_name.c_str(), (int)tensor_type, (int)device);
    }

    ~ggml_qnn_tensor() { _qnn_rpc_buffer.reset(); }

    bool alloc_qnn_tensor_id() {
        if (QNN_TENSOR_GET_ID(_qnn_tensor)) {
            QNN_LOG_WARN("graph tensor %s already created, id %d", _tensor_name.c_str(),
                         QNN_TENSOR_GET_ID(_qnn_tensor));
            return true;
        }

        Qnn_Tensor_t qnn_tensor = _qnn_tensor;
        auto qnn_interface = _qnn_instance->get_qnn_interface();
        auto error = qnn_interface->qnn_tensor_create_graph_tensor(_graph_handle, &qnn_tensor);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("create graph tensor failed, tensor %s, error: %d\n", _tensor_name.c_str(), error);
            return false;
        }

        QNN_TENSOR_SET_ID(_qnn_tensor, QNN_TENSOR_GET_ID(qnn_tensor));
        QNN_LOG_DEBUG("create graph tensor %s, id: %d, rank: %d", _tensor_name.c_str(), QNN_TENSOR_GET_ID(qnn_tensor),
                      QNN_TENSOR_GET_RANK(qnn_tensor));

        return true;
    }

    bool bind_buffer(uint8_t *buffer, const size_t buffer_size) {
        if (_buffer) {
            if (_buffer != buffer) {
                QNN_LOG_WARN("tensor %s has been bound to another buffer %p", _tensor_name.c_str(), _buffer);
                return false;
            }

            QNN_LOG_INFO("tensor %s already bound to same ggml tensor %p", _tensor_name.c_str(), _buffer);
            return true;
        }

        if (QNN_TENSOR_GET_TYPE(_qnn_tensor) == QNN_TENSOR_TYPE_NATIVE) {
            QNN_LOG_DEBUG("tensor %s type(%d) not READ/WRITE, skipping", _tensor_name.c_str(),
                          (int)QNN_TENSOR_TYPE_NATIVE);
            return true;
        }

        if (should_use_mem_handle()) {
            if (!_qnn_rpc_buffer) {
                auto qnn_rpc_buffer = std::make_unique<ggml_qnn_rpc_buffer>(
                    _qnn_instance, buffer_size, QNN_TENSOR_GET_RANK(_qnn_tensor),
                    QNN_TENSOR_GET_DIMENSIONS(_qnn_tensor), QNN_TENSOR_GET_DATA_TYPE(_qnn_tensor));
                if (!qnn_rpc_buffer->is_valid()) {
                    QNN_LOG_WARN("alloc rpc mem failed, tensor %s", _tensor_name.c_str());
                    return false;
                }

                _qnn_rpc_buffer = std::move(qnn_rpc_buffer);
            }

            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(_qnn_tensor, _qnn_rpc_buffer->get_mem_handle());
            QNN_LOG_DEBUG("tensor %s, use mem handle %p", _tensor_name.c_str(), QNN_TENSOR_GET_MEM_HANDLE(_qnn_tensor));
        } else {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = { buffer, (uint32_t)buffer_size };
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("tensor %s, use client buffer %p size %d", _tensor_name.c_str(), client_buf.data,
                          (int)client_buf.dataSize);
        }

        _buffer = buffer;
        _buffer_size = buffer_size;

        if (!write_to_qnn_tensor()) {
            QNN_LOG_WARN("write to qnn tensor failed, tensor %s", _tensor_name.c_str());
            return false;
        }

        QNN_LOG_DEBUG("bind tensor %s to buffer: %p, size: %d", _tensor_name.c_str(), buffer, (int)buffer_size);
        return true;
    }

    bool bind_ggml_tensor(ggml_tensor *tensor) {
        if (!bind_buffer(reinterpret_cast<uint8_t *>(tensor->data), ggml_nbytes(tensor))) {
            QNN_LOG_WARN("Failed to bind tensor: %s to ggml tensor: %s", _tensor_name.c_str(), ggml_get_name(tensor));
            return false;
        }

        QNN_LOG_DEBUG("Bind tensor %s to ggml tensor %s", _tensor_name.c_str(), ggml_get_name(tensor));
        return true;
    }

    bool unbind() {
        if (!_graph_handle) {
            QNN_LOG_WARN("tensor %s not bound to any graph", _tensor_name.c_str());
            return false;
        }

        if (!_buffer) {
            QNN_LOG_DEBUG("tensor %s not bound to ggml tensor", _tensor_name.c_str());
            return true;
        }

        if (!read_from_qnn_tensor()) {
            QNN_LOG_WARN("read from qnn tensor failed, tensor %s", _tensor_name.c_str());
            return false;
        }

        if (!should_use_mem_handle()) {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = {};
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("tensor %s, clear client buffer", _tensor_name.c_str());
        }

        QNN_LOG_DEBUG("unbind tensor: %s from buffer: %p, size: %d", _tensor_name.c_str(), _buffer, (int)_buffer_size);
        _buffer = nullptr;
        _buffer_size = 0;
        return true;
    }

    const Qnn_Tensor_t &get_qnn_tensor() const { return _qnn_tensor; }

private:
    bool write_to_qnn_tensor() {
        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_WRITE && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_DEBUG("tensor %s type(%d) not WRITE", _tensor_name.c_str(), (int)tensor_type);
            return true;
        }

        if (should_use_mem_handle()) {
            if (_qnn_rpc_buffer) {
                memcpy(_qnn_rpc_buffer->get_buffer(), _buffer, _buffer_size);
            } else {
                QNN_LOG_WARN("tensor %s: can't find rpcmem from qnn mem handle\n", _tensor_name.c_str());
                return false;
            }
        }

        // For CPU and GPU, the data is already in the tensor.
        QNN_LOG_DEBUG("write tensor %s to qnn", _tensor_name.c_str());
        return true;
    }

    bool read_from_qnn_tensor() {
        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_READ && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_DEBUG("tensor %s type(%d) not READ", _tensor_name.c_str(), (int)tensor_type);
            return true;
        }

        if (should_use_mem_handle()) {
            if (_qnn_rpc_buffer) {
                memcpy(_buffer, _qnn_rpc_buffer->get_buffer(), _buffer_size);
            } else {
                QNN_LOG_WARN("can't find rpcmem from qnn mem handle\n");
                return false;
            }
        }

        // For CPU and GPU, the data is already in the tensor.
        QNN_LOG_DEBUG("read tensor %s from qnn", _tensor_name.c_str());
        return true;
    }

    void update_params_from_ggml_tensor(tensor_type_t tensor_type, const ggml_qnn_dimension_array_t &dimensions,
                                        ggml_type data_type, int rank) {
        GGML_ASSERT(rank <= GGML_MAX_DIMS && rank > 0);
        switch (rank) {
            case 4:
                _dimensions[3] = (uint32_t)(dimensions[3] ? dimensions[3] : 1);
                // fall through
            case 3:
                _dimensions[2] = (uint32_t)(dimensions[2] ? dimensions[2] : 1);
                // fall through
            case 2:
                _dimensions[1] = (uint32_t)(dimensions[1] ? dimensions[1] : 1);
                // fall through
            case 1:
                _dimensions[0] = (uint32_t)dimensions[0];
                break;
        }
        QNN_TENSOR_SET_DATA_TYPE(_qnn_tensor, device_datatype_from_ggml_datatype(data_type));

        // TODO: set the quantizeParams base on the tensor type

        QNN_TENSOR_SET_RANK(_qnn_tensor, (uint32_t)rank);
        QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t client_buf = {};
        QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);

        Qnn_TensorType_t new_tensor_type;
        switch (tensor_type) {
            case INPUT:
                new_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
                break;
            case OUTPUT:
                new_tensor_type = QNN_TENSOR_TYPE_APP_READ;
                break;
            case PARAMETER:
                new_tensor_type = QNN_TENSOR_TYPE_STATIC;
                break;
            default:
                new_tensor_type = QNN_TENSOR_TYPE_NATIVE;
                break;
        }
        QNN_TENSOR_SET_TYPE(_qnn_tensor, new_tensor_type);
        QNN_LOG_INFO("tensor %s changed to type %d", _tensor_name.c_str(), new_tensor_type);
    }

    bool should_use_mem_handle() const {
        return _device == QNN_BACKEND_NPU && QNN_TENSOR_GET_TYPE(_qnn_tensor) != QNN_TENSOR_TYPE_STATIC;
    }

    std::string _tensor_name;
    uint8_t *_buffer = nullptr;
    size_t _buffer_size = 0;
    QNNBackend _device;
    std::shared_ptr<qnn_instance> _qnn_instance;
    Qnn_Tensor_t _qnn_tensor = qnn_tensor_init(kDefaultQnnTensorVersion);
    std::array<uint32_t, GGML_MAX_DIMS> _dimensions = {};
    Qnn_GraphHandle_t _graph_handle = nullptr;
    std::unique_ptr<ggml_qnn_rpc_buffer> _qnn_rpc_buffer;

    DISABLE_COPY(ggml_qnn_tensor);
    DISABLE_MOVE(ggml_qnn_tensor);
};

} // namespace qnn
