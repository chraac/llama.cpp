#include "buffer.hpp"

#include "device.hpp"
#include "tensor.hpp"

namespace {

static hexagon::npu_buffer * get_buffer_object(ggml_backend_buffer_t buffer) {
    return reinterpret_cast<hexagon::npu_buffer *>(buffer->context);
}

static hexagon::npu_buffer_type * get_buffer_type_object(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<hexagon::npu_buffer_type *>(buft->context);
}

void backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete get_buffer_object(buffer);
}

void * backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    return get_buffer_object(buffer)->get_buffer();
}

ggml_status backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * device_object = get_buffer_type_object(buffer->buft)->get_device();
    auto   tensor_object = get_buffer_object(buffer)->init_tensor(tensor, device_object->get_device_handle());
    if (!tensor_object) {
        LOG_ERROR("Failed to init tensor\n");
        return GGML_STATUS_ALLOC_FAILED;
    }

    tensor->extra = tensor_object.get();
    return GGML_STATUS_SUCCESS;
}

void backend_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                               size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

void backend_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                               size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

bool backend_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

void backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * obj = get_buffer_object(buffer);
    memset(obj->get_buffer(), value, obj->get_size());
}

constexpr const ggml_backend_buffer_i backend_buffer_interface = {
    /* .free_buffer     = */ backend_buffer_free_buffer,
    /* .get_base        = */ backend_buffer_get_base,
    /* .init_tensor     = */ backend_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ backend_buffer_set_tensor,
    /* .get_tensor      = */ backend_buffer_get_tensor,
    /* .cpy_tensor      = */ backend_buffer_cpy_tensor,
    /* .clear           = */ backend_buffer_clear,
    /* .reset           = */ nullptr,
};

const char * backend_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return get_buffer_type_object(buft)->get_name();
}

ggml_backend_buffer_t backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return get_buffer_type_object(buft)->allocate_buffer(size);
}

size_t backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return get_buffer_type_object(buft)->get_buffer_alignment();
}

size_t backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return get_buffer_type_object(buft)->get_max_buffer_size();
}

bool backend_buffer_is_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == backend_buffer_type_get_name;
}

}  // namespace

namespace hexagon {

npu_buffer::npu_buffer(common::rpc_mem_ptr allocator, size_t size) : _allocator(allocator), _size(size) {
    constexpr const uint32_t kRpcMemDefaultFlags  = 1;
    constexpr const int      kRpcMemDefaultHeapId = 25;

    if (!_allocator->is_valid()) {
        LOG_ERROR("rpc memory not initialized\n");
        return;
    }

    if (size > _allocator->get_max_alloc_size()) {
        LOG_ERROR("rpc memory size %zu exceeds max alloc size %zu\n", size, _allocator->get_max_alloc_size());
        return;
    }

    _data = _allocator->alloc(kRpcMemDefaultHeapId, kRpcMemDefaultFlags,
                              size);  // TODO: should we use a different flag?
    if (!_data) {
        LOG_ERROR("failed to allocate rpc memory, size: %d MB\n", (int) (size / (1 << 20)));
        return;
    }
}

npu_buffer::~npu_buffer() {
    _allocator->free(_data);
}

std::unique_ptr<npu_tensor> npu_buffer::init_tensor(ggml_tensor * tensor, remote_handle64 device_handle) {
    if (!_data) {
        LOG_ERROR("failed to init tensor, rpc memory not initialized\n");
        return std::unique_ptr<npu_tensor>();
    }

    auto tensor_object = std::make_unique<npu_tensor>(
        tensor, _allocator->to_fd(_data),
        (uint64_t) (reinterpret_cast<uint8_t *>(tensor->data) - reinterpret_cast<uint8_t *>(_data)), device_handle);
}

npu_buffer_type::npu_buffer_type(ggml_backend_dev_t dev, const std::string & name, common::rpc_mem_ptr rpc_mem) :
    _name(name),
    _rpc_mem(rpc_mem) {
    iface = {
        /* .get_name       = */ backend_buffer_type_get_name,
        /* .alloc_buffer   = */ backend_buffer_type_alloc_buffer,
        /* .get_alignment  = */ backend_buffer_type_get_alignment,
        /* .get_max_size   = */ backend_buffer_type_get_max_size,
        /* .get_alloc_size = */ nullptr,  // defaults to ggml_nbytes
        /* .is_host = */ backend_buffer_is_host,
    };
    device  = dev;
    context = this;

    _device = reinterpret_cast<npu_device *>(device->context);
}

size_t npu_buffer_type::get_max_buffer_size() const {
    if (!_rpc_mem) {
        LOG_ERROR("rpc memory not initialized\n");
        return 0;
    }

    return _rpc_mem->get_max_alloc_size();
}

ggml_backend_buffer_t npu_buffer_type::allocate_buffer(size_t size) {
    if (!_rpc_mem) {
        LOG_ERROR("rpc memory not initialized\n");
        return nullptr;
    }

    auto * buffer = new npu_buffer(_rpc_mem, size);
    if (!buffer->is_valid()) {
        delete buffer;
        LOG_ERROR("Failed to allocate buffer of size %zu\n", size);
        return nullptr;
    }

    return ggml_backend_buffer_init(this, backend_buffer_interface, buffer, size);
}

}  // namespace hexagon
