
#include "common.hpp"

namespace {}  // namespace

backend_device_proxy_ptr create_hexagon_backend_context(backend_index_type device) {
    if (device < QNN_BACKEND_COUNT || device >= TOTAL_BACKEND_COUNT) {
        return backend_device_proxy_ptr();
    }

    // TODO: implement hexagon backend context creation
    return backend_device_proxy_ptr();
}
