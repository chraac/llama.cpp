#pragma once

#include "hexagon_npu.h"
#include "tensor.hpp"

namespace hexagon {

class graph {
  public:
    // TODO: add execute direction here
    explicit graph() noexcept {}

    ~graph() noexcept {
        if (_tensors) {
            delete[] _tensors;
        }
    }

    void set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count);

    bool compute();

  private:
    tensor ** _tensors      = nullptr;
    size_t    _tensor_count = 0;

    DISABLE_COPY(graph);
    DISABLE_MOVE(graph);
};

}  // namespace hexagon
