#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "ggml-qnn.h"
#include "ggml.h"
#include "logger.hpp"
#include "QnnTypes.h"

#define QNN_TENSOR_VER(x) ((x).v1)

#ifndef GGML_QNN_ENABLE_PERFORMANCE_TRACKING
#    ifdef NDEBUG
#        define GGML_QNN_ENABLE_PERFORMANCE_TRACKING 0  // enable/disable op's perf log
#    else
#        define GGML_QNN_ENABLE_PERFORMANCE_TRACKING 1  // enable/disable op's perf log
#    endif
#endif

namespace qnn {

using ggml_dimension_array_t = int64_t[GGML_MAX_DIMS];
using ggml_stride_array_t    = size_t[GGML_MAX_DIMS];
using qnn_dimension_array_t  = std::array<uint32_t, GGML_MAX_DIMS>;

qnn_dimension_array_t get_internal_dimension(const ggml_dimension_array_t & dims, uint32_t rank);
qnn_dimension_array_t get_view_internal_dimension(const ggml_tensor * tensor, size_t & element_offser_out);

uint32_t     get_ggml_tensor_rank(const ggml_tensor * tensor);
const char * get_ggml_type_name(ggml_type type);
const char * get_backend_name(QNNBackend device_index);
const char * get_chipset_desc(uint32_t chipset_id);
const char * get_htparch_desc(size_t htp_arch);
intptr_t     align_to(size_t alignment, intptr_t offset);
uint32_t     get_ggml_tensor_data_size(const ggml_tensor * tensor);
const char * get_qnn_tensor_type_name(Qnn_TensorType_t type);

void * page_align_alloc(size_t size);
void   align_free(void * ptr);

const char * opname_from_ggmlop(enum ggml_op ggmlop);

const char * get_qnn_error_string(Qnn_ErrorHandle_t error);

constexpr const Qnn_TensorVersion_t kDefaultQnnTensorVersion = QNN_TENSOR_VERSION_1;

inline Qnn_Tensor_t qnn_tensor_init(Qnn_TensorVersion_t version) {
    Qnn_Tensor_t tensor;
    tensor.version = version;
    if (version == QNN_TENSOR_VERSION_1) {
        tensor.v1 = QNN_TENSOR_V1_INIT;
    } else if (version == QNN_TENSOR_VERSION_2) {
        tensor.v2 = QNN_TENSOR_V2_INIT;
    }
    return tensor;
}

inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).id;
    }

    return 0u;
}

inline const char * get_qnn_tensorname(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).name;
    }
    return nullptr;
}

inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).type;
    }
    return QNN_TENSOR_TYPE_UNDEFINED;
}

inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}

inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).quantizeParams;
    }
    return QNN_QUANTIZE_PARAMS_INIT;
}

inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).rank;
    }
    return 0u;
}

inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).dimensions;
    }
    return nullptr;
}

inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).memType;
    }
    return QNN_TENSORMEMTYPE_UNDEFINED;
}

inline Qnn_MemHandle_t get_qnn_tensor_memhandle(const Qnn_Tensor_t & tensor) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        return QNN_TENSOR_VER(tensor).memHandle;
    }
    return nullptr;
}

inline void set_qnn_tensor_id(Qnn_Tensor_t & tensor, uint32_t id) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).id = id;
    }
}

inline void set_qnn_tensor_name(Qnn_Tensor_t & tensor, const char * name) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).name = name;
    }
}

inline void set_qnn_tensor_type(Qnn_Tensor_t & tensor, Qnn_TensorType_t type) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).type = type;
    }
}

inline void set_qnn_tensor_dataformat(Qnn_Tensor_t & tensor, Qnn_TensorDataFormat_t format) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).dataFormat = format;
    }
}

inline void set_qnn_tensor_datatype(Qnn_Tensor_t & tensor, Qnn_DataType_t dataType) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).dataType = dataType;
    }
}

inline void set_qnn_tensor_quantparams(Qnn_Tensor_t & tensor, Qnn_QuantizeParams_t params) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).quantizeParams = params;
    }
}

inline void set_qnn_tensor_rank(Qnn_Tensor_t & tensor, uint32_t rank) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).rank = rank;
    }
}

inline void set_qnn_tensor_dimensions(Qnn_Tensor_t & tensor, uint32_t * dims) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).dimensions = dims;
    }
}

inline void set_qnn_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t mem_type) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).memType = mem_type;
    }
}

inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t client_buf) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).clientBuf = client_buf;
    }
}

inline void set_qnn_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == kDefaultQnnTensorVersion) {
        QNN_TENSOR_VER(tensor).memHandle = handle;
    }
}

inline void set_qnn_tensor_dyn_dimensions(Qnn_Tensor_t & tensor, uint8_t * isDynamicDimensions) {
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        tensor.v2.isDynamicDimensions = isDynamicDimensions;
    }
}

Qnn_DataType_t qnn_datatype_from_ggml_datatype(ggml_type ggml_type);
ggml_type      ggml_datatype_from_qnn_datatype(Qnn_DataType_t qnn_type);
size_t         qnn_datatype_size(Qnn_DataType_t qnn_type);
const char *   qnn_datatype_to_string(Qnn_DataType_t qnn_type);
size_t         get_system_total_memory_in_bytes();
size_t         get_system_free_memory_in_bytes();

#if GGML_QNN_ENABLE_PERFORMANCE_TRACKING

class qnn_scoped_timer {
  public:
    qnn_scoped_timer(const std::string & log_prefix) : _log_prefix(std::move(log_prefix)) {
        _begin_us = ggml_time_us();
    }

    qnn_scoped_timer(qnn_scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    ~qnn_scoped_timer() { print(); }

    void operator=(qnn_scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    void print() const {
        auto duration = (ggml_time_us() - _begin_us) / 1000.0;
        QNN_LOG_INFO("[profiler]%s, duration: %.4f ms\n", _log_prefix.c_str(), duration);
    }


  private:
    int64_t     _begin_us = 0LL;
    std::string _log_prefix;

    qnn_scoped_timer(const qnn_scoped_timer &) = delete;
    void operator=(const qnn_scoped_timer &)   = delete;
};

inline qnn_scoped_timer make_scope_perf_timer(const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return qnn_scoped_timer(buffer);
}

#else

inline void make_scope_perf_timer(const char *, ...) {}

#endif

}  // namespace qnn

#define QNN_TENSOR_GET_ID(tensor)           qnn::get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)         qnn::get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)         qnn::get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)  qnn::get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)    qnn::get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor) qnn::get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)         qnn::get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)   qnn::get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)     qnn::get_qnn_tensor_memtype(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor)   qnn::get_qnn_tensor_memhandle(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)             qnn::set_qnn_tensor_id(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)           qnn::set_qnn_tensor_name(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)           qnn::set_qnn_tensor_type(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)    qnn::set_qnn_tensor_dataformat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)      qnn::set_qnn_tensor_datatype(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)   qnn::set_qnn_tensor_quantparams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)           qnn::set_qnn_tensor_rank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)     qnn::set_qnn_tensor_dimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)       qnn::set_qnn_tensor_memtype(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)     qnn::set_qnn_tensor_clientbuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)     qnn::set_qnn_tensor_memhandle(tensor, value)
#define QNN_TENSOR_SET_DYN_DIMENSIONS(tensor, value) qnn::set_qnn_tensor_dyn_dimensions(tensor, value)

#if GGML_QNN_ENABLE_PERFORMANCE_TRACKING
#    define QNN_SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto __qnn_timer_##__LINE__ = qnn::make_scope_perf_timer(fmt, ##__VA_ARGS__)
#else
#    define QNN_SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#endif
