#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "logger.hpp"

namespace qnn {

#ifdef GGML_QNN_ENABLE_PERFORMANCE_TRACKING

class qnn_scoped_timer {
  public:
    qnn_scoped_timer(const std::string & log_prefix) : _log_prefix(std::move(log_prefix)) {
        _begin_us = ggml_time_us();
    }

    qnn_scoped_timer(qnn_scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    ~qnn_scoped_timer() { print(); }

    void operator=(qnn_scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    void print() const {
        auto duration = (ggml_time_us() - _begin_us) / 1000.0;
        QNN_LOG_INFO("[profiler]%s, duration: %.4f ms\n", _log_prefix.c_str(), duration);
    }


  private:
    int64_t     _begin_us = 0LL;
    std::string _log_prefix;

    qnn_scoped_timer(const qnn_scoped_timer &) = delete;
    void operator=(const qnn_scoped_timer &)   = delete;
};

inline qnn_scoped_timer make_scope_perf_timer(const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return qnn_scoped_timer(buffer);
}

#else

inline void make_scope_perf_timer(const char *, ...) {}

#endif

}  // namespace qnn

#ifdef GGML_QNN_ENABLE_PERFORMANCE_TRACKING
#    define QNN_SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto __qnn_timer_##__LINE__ = qnn::make_scope_perf_timer(fmt, ##__VA_ARGS__)
#else
#    define QNN_SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#endif
