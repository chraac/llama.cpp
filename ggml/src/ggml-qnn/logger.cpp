
#include "logger.hpp"

#include <cstdio>
#include <mutex>

#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#endif

void qnn::internal_log(ggml_log_level level, const char * /*file*/, const char *func, int line, const char *format,
                       ...) {
    static std::mutex qnn_internal_log_mutex;
    static char s_qnn_internal_log_buf[QNN_LOGBUF_LEN];

    {
        std::lock_guard<std::mutex> lock(qnn_internal_log_mutex);
        va_list args;

        va_start(args, format);
        int len_prefix = snprintf(s_qnn_internal_log_buf, QNN_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_qnn_internal_log_buf + len_prefix, QNN_LOGBUF_LEN - len_prefix, format, args);
        if (len < (QNN_LOGBUF_LEN - len_prefix)) {
#if defined(__ANDROID__) || defined(ANDROID)
            // print to android logcat
            __android_log_print(level, "ggml-qnn", "%s\n", s_qnn_internal_log_buf);
#else
            (void)level;
#endif
            // print to stdout
            printf("%s\n", s_qnn_internal_log_buf);
        }
        va_end(args);
    }
}

#if ENABLE_QNNSDK_LOG
void qnn::sdk_logcallback(const char *fmt, QnnLog_Level_t level, uint64_t /*timestamp*/, va_list argp) {
    static std::mutex log_mutex;
    static char s_ggml_qnn_logbuf[QNN_LOGBUF_LEN];

    const char *log_level_desc = "";
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
        vsnprintf(s_ggml_qnn_logbuf, QNN_LOGBUF_LEN, fmt, argp);
        QNN_LOG_INFO("[%s]%s", log_level_desc, s_ggml_qnn_logbuf);
    }
}
#else
void qnn::sdk_logcallback(const char *, QnnLog_Level_t, uint64_t, va_list) {}
#endif
