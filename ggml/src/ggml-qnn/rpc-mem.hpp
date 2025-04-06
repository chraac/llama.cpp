
#pragma once

#include <limits>

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
    using rpc_mem_init_t   = void (*)();
    using rpc_mem_deinit_t = void (*)();
    using rpc_mem_alloc_t  = void * (*) (int heapid, uint32_t flags, int size);
    using rpc_mem_alloc2_t = void * (*) (int heapid, uint32_t flags, size_t size);
    using rpc_mem_free_t   = void (*)(void * po);
    using rpc_mem_to_fd_t  = int (*)(void * po);

    rpc_mem(const std::string & rpc_lib_path = kQnnRpcLibName) {
        _rpc_lib_handle = dl_load(rpc_lib_path);
        if (!_rpc_lib_handle) {
            LOG_ERROR("failed to load %s, error: %s\n", rpc_lib_path.c_str(), dl_error());
            return;
        }

        _rpc_mem_init   = reinterpret_cast<rpc_mem_init_t>(dl_sym(_rpc_lib_handle, "rpcmem_init"));
        _rpc_mem_deinit = reinterpret_cast<rpc_mem_deinit_t>(dl_sym(_rpc_lib_handle, "rpcmem_deinit"));
        _rpc_mem_alloc  = reinterpret_cast<rpc_mem_alloc_t>(dl_sym(_rpc_lib_handle, "rpcmem_alloc"));
        _rpc_mem_alloc2 = reinterpret_cast<rpc_mem_alloc2_t>(dl_sym(_rpc_lib_handle, "rpcmem_alloc2"));
        _rpc_mem_free   = reinterpret_cast<rpc_mem_free_t>(dl_sym(_rpc_lib_handle, "rpcmem_free"));
        _rpc_mem_to_fd  = reinterpret_cast<rpc_mem_to_fd_t>(dl_sym(_rpc_lib_handle, "rpcmem_to_fd"));
        if (_rpc_mem_init) {
            _rpc_mem_init();
        }

#ifdef NDEBUG
        if (_rpc_mem_alloc2) {
            LOG_DEBUG("rpcmem alloc2 is supported\n");
        } else {
            LOG_DEBUG("rpcmem alloc2 is not supported\n");
        }
#endif

        LOG_DEBUG("load rpcmem lib successfully\n");
    }

    ~rpc_mem() {
        if (_rpc_lib_handle) {
            if (_rpc_mem_deinit) {
                _rpc_mem_deinit();
            }

            dl_unload(_rpc_lib_handle);
        }

        LOG_DEBUG("unload rpcmem lib successfully\n");
    }

    bool is_valid() const { return _rpc_lib_handle != nullptr; }

    void * alloc(int heapid, uint32_t flags, size_t size) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
            return nullptr;
        }

        if (size > get_max_alloc_size()) {
            LOG_ERROR("rpc memory size %zu exceeds max alloc size %zu\n", size, get_max_alloc_size());
            return nullptr;
        }

        void * buf = nullptr;
        if (_rpc_mem_alloc2) {
            buf = _rpc_mem_alloc2(heapid, flags, size);
        } else {
            buf = _rpc_mem_alloc(heapid, flags, size);
        }

        if (!buf) {
            LOG_ERROR("failed to allocate rpc memory, size: %d MB\n", (int) (size / (1 << 20)));
            return nullptr;
        }

        return buf;
    }

    void free(void * buf) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
        } else {
            _rpc_mem_free(buf);
        }
    }

    int to_fd(void * buf) {
        int mem_fd = -1;
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
        } else {
            mem_fd = _rpc_mem_to_fd(buf);
        }

        return mem_fd;
    }

    size_t get_max_alloc_size() {
        return _rpc_mem_alloc2 ? std::numeric_limits<size_t>::max() : std::numeric_limits<int>::max();
    }

  private:
    dl_handler_t     _rpc_lib_handle = nullptr;
    rpc_mem_init_t   _rpc_mem_init   = nullptr;
    rpc_mem_deinit_t _rpc_mem_deinit = nullptr;
    rpc_mem_alloc_t  _rpc_mem_alloc  = nullptr;
    rpc_mem_alloc2_t _rpc_mem_alloc2 = nullptr;
    rpc_mem_free_t   _rpc_mem_free   = nullptr;
    rpc_mem_to_fd_t  _rpc_mem_to_fd  = nullptr;
};

}  // namespace common
