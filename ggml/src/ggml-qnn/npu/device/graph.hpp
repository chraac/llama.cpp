#pragma once

#include <memory>

#include "hexagon_npu.h"
#include "tensor.hpp"

namespace hexagon {

class graph {
  public:
    // TODO: add execute direction here
    explicit graph() noexcept;

    ~graph() noexcept;

    void set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count);

    bool compute();

  private:
    std::unique_ptr<tensor *[]> _tensors;
    size_t                      _tensor_count = 0;

    DISABLE_COPY(graph);
    DISABLE_MOVE(graph);
};

}  // namespace hexagon
