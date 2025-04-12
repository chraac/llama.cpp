
#include "graph.hpp"

#include "op_impl.hpp"
#include "util.hpp"

namespace hexagon {

graph::~graph() noexcept {
    if (_tensors) {
        delete[] _tensors;
    }
}

void graph::set_tensor(const npu_device_tensor_handle_t * tensors, int tensor_count) {
    _tensors = new tensor *[tensor_count];
    for (int i = 0; i < tensor_count; ++i) {
        _tensors[i] = reinterpret_cast<tensor *>(tensors[i]);
        DEVICE_LOG_DEBUG("graph(%p) tensor[%d]: %p\n", (void *) this, i, (void *) _tensors[i]);
    }

    _tensor_count = tensor_count;
    DEVICE_LOG_DEBUG("graph(%p) tensor count: %zu\n", (void *) this, _tensor_count);
}

bool graph::compute() {
    if (!_tensors) {
        return false;
    }

    DEVICE_LOG_DEBUG("graph(%p) compute\n", (void *) this);
    for (size_t i = 0; i < _tensor_count; ++i) {
        auto * op   = _tensors[i];
        auto * func = get_compute_func(op->get_op());
        func(op);
    }

    return true;
}

}  // namespace hexagon
