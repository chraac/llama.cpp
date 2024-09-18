
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "ggml-qnn.h"

#include "buffer.hpp"
#include "logger.hpp"
#include "qnn-lib.hpp"
#include "utils.hpp"

namespace qnn {

using ggml_tensor_array_t = std::vector<ggml_tensor *>;

class ggml_qnn_tensor {
public:
    explicit ggml_qnn_tensor(const std::string &name, QNNBackend device, Qnn_GraphHandle_t graph_handle,
                             std::shared_ptr<qnn_instance> qnn_instance) :
        _tensor_name(name), _device(device), _qnn_instance(qnn_instance), _graph_handle(graph_handle) {
        QNN_TENSOR_SET_NAME(_qnn_tensor, _tensor_name.c_str());
        QNN_TENSOR_SET_DIMENSIONS(_qnn_tensor, _dimensions.data());
        QNN_TENSOR_SET_TYPE(_qnn_tensor, QNN_TENSOR_TYPE_NATIVE);
        QNN_TENSOR_SET_DATA_FORMAT(_qnn_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
        QNN_LOG_DEBUG("create tensor %s, device: %d", _tensor_name.c_str(), device);
    }

    ~ggml_qnn_tensor() { _qnn_rpc_buffer.reset(); }

    bool create_tensor(const ggml_tensor *tensor, bool is_input, int prev_max_rank) {
        update_params_from_ggml_tensor(tensor, is_input, prev_max_rank);

        if (!QNN_TENSOR_GET_ID(_qnn_tensor)) {
            Qnn_Tensor_t qnn_tensor = _qnn_tensor;
            auto qnn_interface = _qnn_instance->get_qnn_interface();
            auto error = qnn_interface->qnn_tensor_create_graph_tensor(_graph_handle, &qnn_tensor);
            if (error != QNN_SUCCESS) {
                QNN_LOG_WARN("create graph tensor failed, tensor %s, error: %d\n", _tensor_name.c_str(), error);
                return false;
            }

            QNN_TENSOR_SET_ID(_qnn_tensor, QNN_TENSOR_GET_ID(qnn_tensor));
            QNN_LOG_DEBUG("create graph tensor %s, id: %d, rank: %d", _tensor_name.c_str(),
                          QNN_TENSOR_GET_ID(qnn_tensor), QNN_TENSOR_GET_RANK(qnn_tensor));
        }

        return true;
    }

    bool bind_ggml_tensor(ggml_tensor *tensor) {
        if (_tensor) {
            if (_tensor != tensor) {
                QNN_LOG_WARN("tensor %s has been bound to another ggml tensor %s", _tensor_name.c_str(),
                             ggml_get_name(_tensor));
                return false;
            }
            QNN_LOG_INFO("tensor %s already bound to same ggml tensor %s", _tensor_name.c_str(),
                         ggml_get_name(_tensor));
            return true;
        }

        if (should_use_mem_handle()) {
            if (!_qnn_rpc_buffer) {
                auto qnn_rpc_buffer = std::make_unique<ggml_qnn_rpc_buffer>(
                    _qnn_instance, ggml_nbytes(tensor), QNN_TENSOR_GET_RANK(_qnn_tensor),
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
            Qnn_ClientBuffer_t client_buf = { tensor->data, get_ggml_tensor_data_size(tensor) };
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("tensor %s, use client buffer %p size %d", _tensor_name.c_str(), client_buf.data,
                          (int)client_buf.dataSize);
        }

        _tensor = tensor;

        if (!write_to_qnn_tensor()) {
            QNN_LOG_WARN("write to qnn tensor failed, tensor %s", _tensor_name.c_str());
            return false;
        }

        QNN_LOG_DEBUG("bind tensor %s to ggml tensor %s", _tensor_name.c_str(), ggml_get_name(tensor));
        return true;
    }

    bool unbind_ggml_tensor() {
        if (!_graph_handle) {
            QNN_LOG_WARN("tensor %s not bound to any graph", _tensor_name.c_str());
            return false;
        }

        if (!_tensor) {
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

        QNN_LOG_DEBUG("unbind tensor: %s from ggml tensor: %s", _tensor_name.c_str(), ggml_get_name(_tensor));
        _tensor = nullptr;
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
                memcpy(_qnn_rpc_buffer->get_buffer(), _tensor->data, ggml_nbytes(_tensor));
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
                memcpy(_tensor->data, _qnn_rpc_buffer->get_buffer(), ggml_nbytes(_tensor));
            } else {
                QNN_LOG_WARN("can't find rpcmem from qnn mem handle\n");
                return false;
            }
        }

        // For CPU and GPU, the data is already in the tensor.
        QNN_LOG_DEBUG("read tensor %s from qnn", _tensor_name.c_str());
        return true;
    }

    void update_params_from_ggml_tensor(const ggml_tensor *tensor, bool is_input, int prev_max_rank) {
        _dimensions[0] = (uint32_t)tensor->ne[0];
        _dimensions[1] = (uint32_t)tensor->ne[1];
        _dimensions[2] = (uint32_t)tensor->ne[2];
        _dimensions[3] = (uint32_t)tensor->ne[3];
        QNN_TENSOR_SET_DATA_TYPE(_qnn_tensor, device_datatype_from_ggml_datatype(tensor->type));
        // TODO: set the quantizeParams base on the tensor type

        QNN_TENSOR_SET_RANK(_qnn_tensor, (uint32_t)std::max(prev_max_rank, ggml_n_dims(tensor)));

        QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t client_buf = {};
        QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);

        Qnn_TensorType_t new_tensor_type = is_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
        QNN_TENSOR_SET_TYPE(_qnn_tensor, new_tensor_type);

        QNN_LOG_INFO("tensor %s changed to type %d", _tensor_name.c_str(), new_tensor_type);
    }

    bool should_use_mem_handle() const { return _device == QNN_BACKEND_NPU; }

    std::string _tensor_name;
    const ggml_tensor *_tensor;
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
