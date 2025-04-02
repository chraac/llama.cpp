#pragma once

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

    void set_tensor(const remote_handle64 * tensors, size_t tensor_count);

    bool compute();

  private:
    Tensor ** _tensors      = nullptr;
    size_t    _tensor_count = 0;

    graph(const graph &)             = delete;
    graph & operator=(const graph &) = delete;
};

}  // namespace hexagon
