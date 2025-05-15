#pragma once

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <cstdint>
#include <cstring>

#include "hexagon_npu.h"

#define DEVICE_LOG_ERROR(...) FARF(FATAL, __VA_ARGS__)
#define DEVICE_LOG_WARN(...)  FARF(ERROR, __VA_ARGS__)
#define DEVICE_LOG_INFO(...)  FARF(HIGH, __VA_ARGS__)

#ifdef _DEBUG
#    undef FARF_LOW
#    define FARF_LOW              1
#    define DEVICE_LOG_DEBUG(...) FARF(LOW, __VA_ARGS__)
#else
#    define DEVICE_LOG_DEBUG(...) (void) 0
#endif

// TODO: reuse the declaration at host
#define DISABLE_COPY(class_name)                 \
    class_name(const class_name &)     = delete; \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)            \
    class_name(class_name &&)     = delete; \
    void operator=(class_name &&) = delete

#define DISABLE_COPY_AND_MOVE(class_name) \
    DISABLE_COPY(class_name);             \
    DISABLE_MOVE(class_name)

#define NPU_UNUSED(x) (void) (x)

namespace hexagon {

inline constexpr const char * op_get_name(npu_device_tensor_op op) {
    switch (op) {
        case NPU_OP_MUL_MAT:
            return "MUL_MAT";
        case NPU_OP_ADD:
            return "ADD";
        case NPU_OP_SUB:
            return "SUB";
        case NPU_OP_MUL:
            return "MUL";
        default:
            return "UNKNOWN";
    }
}

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING

template <size_t _buffer_count> class npu_scoped_timer {
  public:
    explicit npu_scoped_timer(const char * log_prefix) {
        strncpy(_log_prefix, log_prefix, _buffer_count - 1);
        _begin_cycles = HAP_perf_get_qtimer_count();
    }

    npu_scoped_timer(npu_scoped_timer && other) {
        strncpy(_log_prefix, other._log_prefix, _buffer_count - 1);
        _begin_cycles = other._begin_cycles;
    }

    ~npu_scoped_timer() { print(); }

    void operator=(npu_scoped_timer && other) {
        strncpy(_log_prefix, other._log_prefix, _buffer_count - 1);
        _begin_cycles = other._begin_cycles;
    }

    void print() const {
        auto total_cycles = HAP_perf_get_qtimer_count() - _begin_cycles;
        auto duration     = HAP_perf_qtimer_count_to_us(total_cycles);
        DEVICE_LOG_WARN("[profiler]%s, cyc: %llu, dur: %lluus\n", _log_prefix, total_cycles, duration);
    }

  private:
    char     _log_prefix[_buffer_count] = {};
    uint64_t _begin_cycles              = 0;

    DISABLE_COPY(npu_scoped_timer);
};

inline auto make_scoped_perf_timer(const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return npu_scoped_timer<1024>(buffer);
}

#endif

}  // namespace hexagon

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
#    define DEVICE_SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto __npu_timer_##__LINE__ = hexagon::make_scoped_perf_timer(fmt, __VA_ARGS__)
#else
#    define DEVICE_SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#endif
