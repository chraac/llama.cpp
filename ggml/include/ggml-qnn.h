#pragma once

#include "ggml.h"

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_QNN_NAME "QNN"
#define GGML_QNN_MAX_DEVICES 3

enum QNNBackend {
    QNN_BACKEND_CPU = 0,
    QNN_BACKEND_GPU,
    QNN_BACKEND_NPU,
    QNN_BACKEND_GGML, //"fake" QNN backend, used for compare performance between
                      // QNN and original GGML
};

/**
 *
 * @param index                       0: QNN_BACKEND_CPU 1: QNN_BACKEND_GPU 2:QNN_BACKEND_NPU
 * @param extend_lib_search_path      extened lib search path for searching QNN backend dynamic libs
 * @return
 */
GGML_API ggml_backend_t ggml_backend_qnn_init(size_t index, const char *extend_lib_search_path);

GGML_API bool ggml_backend_is_qnn(ggml_backend_t backend);

GGML_API int ggml_backend_qnn_get_device_count(void);

GGML_API ggml_backend_reg_t ggml_backend_qnn_reg(void);

#ifdef __cplusplus
}
#endif
