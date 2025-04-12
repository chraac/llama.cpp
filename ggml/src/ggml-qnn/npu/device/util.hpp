#pragma once

#include <HAP_farf.h>

#define DEVICE_LOG_ERROR(...) FARF(FATAL, __VA_ARGS__)
#define DEVICE_LOG_WARN(...)  FARF(ERROR, __VA_ARGS__)
#define DEVICE_LOG_INFO(...)  FARF(HIGH, __VA_ARGS__)

#undef FARF_LOW
#undef FARF_HIGH
#undef FARF_ERROR
#undef FARF_FATAL

#ifdef _DEBUG
#    define FARF_LOW              1
#    define FARF_HIGH             1
#    define FARF_ERROR            1
#    define FARF_FATAL            1
#    define DEVICE_LOG_DEBUG(...) FARF(LOW, __VA_ARGS__)
#else
#    define FARF_LOW              0
#    define FARF_HIGH             1
#    define FARF_ERROR            1
#    define FARF_FATAL            1
#    define DEVICE_LOG_DEBUG(...) (void) 0
#endif
