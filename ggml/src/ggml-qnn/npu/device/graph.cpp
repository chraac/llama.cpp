
#include "graph.hpp"

#include "op_impl.hpp"

namespace hexagon {

void graph::set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count) {
    _tensors = new tensor *[tensor_count];
    for (int i = 0; i < tensor_count; ++i) {
        _tensors[i] = reinterpret_cast<tensor *>(tensors[i]);
    }
    _tensor_count = tensor_count;
}

bool graph::compute() {
    if (!_tensors) {
        return false;
    }

    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * op   = _tensors[i];
        auto * func = get_compute_func(op->get_op());
        func(op);
    }

    return true;
}

}  // namespace hexagon
