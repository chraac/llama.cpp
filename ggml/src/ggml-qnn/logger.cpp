
#include "logger.hpp"

#ifndef NDEBUG

#    include <mutex>

#    include "QnnInterface.h"
#    include "QnnTypes.h"
#    include "System/QnnSystemInterface.h"

void qnn::sdk_logcallback(const char * fmt, QnnLog_Level_t level, uint64_t /*timestamp*/, va_list argp) {
    static std::mutex log_mutex;
    static char       s_ggml_qnn_logbuf[4096];

    const char * log_level_desc = "";
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            log_level_desc = "ERROR";
            break;
        case QNN_LOG_LEVEL_WARN:
            log_level_desc = "WARNING";
            break;
        case QNN_LOG_LEVEL_INFO:
            log_level_desc = "INFO";
            break;
        case QNN_LOG_LEVEL_DEBUG:
            log_level_desc = "DEBUG";
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            log_level_desc = "VERBOSE";
            break;
        case QNN_LOG_LEVEL_MAX:
            log_level_desc = "UNKNOWN";
            break;
    }

    {
        std::lock_guard<std::mutex> lock(log_mutex);
        vsnprintf(s_ggml_qnn_logbuf, sizeof(s_ggml_qnn_logbuf), fmt, argp);
        QNN_LOG_INFO("[%s]%s\n", log_level_desc, s_ggml_qnn_logbuf);
    }
}
#else
void qnn::sdk_logcallback(const char *, QnnLog_Level_t, uint64_t, va_list) {}
#endif
