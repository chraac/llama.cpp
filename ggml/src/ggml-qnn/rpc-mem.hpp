
#pragma once

#include "common.hpp"
#include "dyn-lib-loader.hpp"

namespace common {

#ifdef _WIN32
constexpr const char * kQnnRpcLibName = "libcdsprpc.dll";
#else
constexpr const char * kQnnRpcLibName = "libcdsprpc.so";
#endif

class rpc_mem {
  public:
    using pfn_rpc_mem_init   = void (*)();
    using pfn_rpc_mem_deinit = void (*)();
    using pfn_rpc_mem_alloc  = void * (*) (int heapid, uint32_t flags, int size);
    using pfn_rpc_mem_alloc2 = void * (*) (int heapid, uint32_t flags, size_t size);
    using pfn_rpc_mem_free   = void (*)(void * po);
    using pfn_rpc_mem_to_fd  = int (*)(void * po);

    rpc_mem(const std::string & rpc_lib_path = kQnnRpcLibName) {
        _rpc_lib_handle = dl_load(rpc_lib_path);
        if (!_rpc_lib_handle) {
            LOG_ERROR("failed to load %s, error: %s\n", rpc_lib_path.c_str(), dl_error());
            return;
        }

        _pfn_rpc_mem_init   = reinterpret_cast<pfn_rpc_mem_init>(dl_sym(_rpc_lib_handle, "rpcmem_init"));
        _pfn_rpc_mem_deinit = reinterpret_cast<pfn_rpc_mem_deinit>(dl_sym(_rpc_lib_handle, "rpcmem_deinit"));
        _pfn_rpc_mem_alloc  = reinterpret_cast<pfn_rpc_mem_alloc>(dl_sym(_rpc_lib_handle, "rpcmem_alloc"));
        _pfn_rpc_mem_alloc2 = reinterpret_cast<pfn_rpc_mem_alloc2>(dl_sym(_rpc_lib_handle, "rpcmem_alloc2"));
        _pfn_rpc_mem_free   = reinterpret_cast<pfn_rpc_mem_free>(dl_sym(_rpc_lib_handle, "rpcmem_free"));
        _pfn_rpc_mem_to_fd  = reinterpret_cast<pfn_rpc_mem_to_fd>(dl_sym(_rpc_lib_handle, "rpcmem_to_fd"));
        if (_pfn_rpc_mem_init) {
            _pfn_rpc_mem_init();
        }

        LOG_DEBUG("load rpcmem lib successfully\n");
    }

    ~rpc_mem() {
        if (_rpc_lib_handle) {
            if (_pfn_rpc_mem_deinit) {
                _pfn_rpc_mem_deinit();
            }

            dl_unload(_rpc_lib_handle);
        }

        LOG_DEBUG("unload rpcmem lib successfully\n");
    }

    bool is_valid() const { return _rpc_lib_handle != nullptr; }

    void * alloc(int heapid, uint32_t flags, int size) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
            return nullptr;
        }

        void * buf = _pfn_rpc_mem_alloc(heapid, flags, size);
        if (!buf) {
            LOG_ERROR("failed to allocate rpc memory, size: %d MB\n", (int) (size / (1 << 20)));
            return nullptr;
        }

        return buf;
    }

    void * alloc2(int heapid, uint32_t flags, size_t size) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
            return nullptr;
        }

        void * buf = _pfn_rpc_mem_alloc2(heapid, flags, size);
        if (!buf) {
            LOG_ERROR("failed to allocate rpc memory, size: %zu MB\n", (size / (1 << 20)));
            return nullptr;
        }

        return buf;
    }

    void free(void * buf) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
        } else {
            _pfn_rpc_mem_free(buf);
        }
    }

    int to_fd(void * buf) {
        int mem_fd = -1;
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
        } else {
            mem_fd = _pfn_rpc_mem_to_fd(buf);
        }

        return mem_fd;
    }

  private:
    dl_handler_t       _rpc_lib_handle     = nullptr;
    pfn_rpc_mem_init   _pfn_rpc_mem_init   = nullptr;
    pfn_rpc_mem_deinit _pfn_rpc_mem_deinit = nullptr;
    pfn_rpc_mem_alloc  _pfn_rpc_mem_alloc  = nullptr;
    pfn_rpc_mem_alloc2 _pfn_rpc_mem_alloc2 = nullptr;
    pfn_rpc_mem_free   _pfn_rpc_mem_free   = nullptr;
    pfn_rpc_mem_to_fd  _pfn_rpc_mem_to_fd  = nullptr;
};

}  // namespace common
