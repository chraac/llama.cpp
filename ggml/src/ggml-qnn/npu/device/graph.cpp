
#include "graph.hpp"

#include "op_impl.hpp"

namespace hexagon {

void graph::set_tensor(const remote_handle64 * tensors, size_t tensor_count) {
    _tensors = new Tensor *[tensor_count];
    for (size_t i = 0; i < tensor_count; ++i) {
        _tensors[i] = reinterpret_cast<Tensor *>(tensors[i]);
    }
    _tensor_count = tensor_count
}

bool graph::compute() {
    if (!_tensors) {
        return false;
    }

    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * op   = _tensors[i];
        auto * func = get_compute_func(op->op);
        func(op);
    }

    return true;
}

}  // namespace hexagon
