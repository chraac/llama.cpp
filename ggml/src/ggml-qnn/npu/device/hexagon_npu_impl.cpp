
#include <hexagon_types.h>

#include "AEEStdErr.h"
#include "graph.hpp"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "remote.h"
#include "tensor.hpp"

namespace {

inline hexagon::tensor * from_handle(remote_handle64 h) {
    return reinterpret_cast<hexagon::tensor *>(h);
}

inline remote_handle64 to_handle(hexagon::tensor * tensor) {
    return reinterpret_cast<remote_handle64>(tensor);
}

inline hexagon::graph * from_handle(remote_handle64 h) {
    return reinterpret_cast<hexagon::graph *>(h);
}

inline remote_handle64 to_handle(hexagon::graph * graph) {
    return reinterpret_cast<remote_handle64>(graph);
}

}  // namespace

AEEResult npu_device_tensor_init(const tensor_info * info, remote_handle64 * h) {
    auto * tensor = new (std::nothrow) hexagon::tensor(*info);
    if (!tensor) {
        return AEE_ENOMEMORY;
    }

    *h = to_handle(tensor);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_set_src(remote_handle64 h, size_t index, remote_handle64 src) {
    auto * tensor = from_handle(h);
    if (!tensor) {
        return AEE_EINVHANDLE;
    }

    auto * src_tensor = from_handle(src);
    tensor->set_src(index, src_tensor);
    return AEE_SUCCESS;
}

AEEResult npu_device_tensor_free(remote_handle64 h) {
    auto * tensor = from_handle(h);
    if (!tensor) {
        return AEE_EINVHANDLE;
    }

    delete tensor;
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_init(remote_handle64 * h) {
    auto * graph = new (std::nothrow) hexagon::graph();
    if (!graph) {
        return AEE_ENOMEMORY;
    }

    *h = to_handle(graph);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_set_tensor(remote_handle64 h, const remote_handle64 * tensors, int tensor_count) {
    auto * graph = from_handle(h);
    if (!graph || !tensors || tensor_count <= 0) {
        return AEE_EINVHANDLE;
    }

    graph->set_tensor(tensors, tensor_count);
    return AEE_SUCCESS;
}

AEEResult npu_device_graph_compute(remote_handle64 h) {
    auto * graph = from_handle(h);
    if (!graph) {
        return AEE_EINVHANDLE;
    }

    if (!graph->compute()) {
        return AEE_EFAILED;
    }

    return AEE_SUCCESS;
}
