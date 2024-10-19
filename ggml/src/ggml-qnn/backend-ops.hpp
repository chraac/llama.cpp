#pragma once

#include "ggml.h"

#include "backend.hpp"

namespace qnn {

bool ggml_qnn_supports_op(ggml_backend_qnn_device_context *ctx, const ggml_tensor *op);
bool ggml_qnn_forward(ggml_backend_qnn_device_context *ctx, struct ggml_tensor *tensor);

} // namespace qnn
