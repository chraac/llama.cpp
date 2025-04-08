#pragma once

#include <vector>

#include "common.hpp"
#include "ggml-backend-impl.h"
#include "hexagon_npu.h"

namespace hexagon {

class npu_graph {
  public:
    npu_graph(ggml_cgraph * cgraph, remote_handle64 device_handle);

    ~npu_graph();

    bool is_valid() const { return _graph_handle != 0; }

    bool compute();

  private:
    remote_handle64                         _device_handle = 0;
    npu_device_graph_handle_t               _graph_handle  = 0;
    std::vector<npu_device_tensor_handle_t> _tensor_handles;

    DISABLE_COPY(npu_graph);
    DISABLE_MOVE(npu_graph);
};

}  // namespace hexagon
