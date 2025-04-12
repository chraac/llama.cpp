#pragma once

#include <HAP_farf.h>

#define DEVICE_LOG_ERROR(...) FARF(FATAL, __VA_ARGS__)
#define DEVICE_LOG_WARN(...)  FARF(ERROR, __VA_ARGS__)
#define DEVICE_LOG_INFO(...)  FARF(HIGH, __VA_ARGS__)

#ifdef _DEBUG
// TODO: check why FARF_LOW is not working
#    define DEVICE_LOG_DEBUG(...) FARF(HIGH, __VA_ARGS__)
#else
#    define DEVICE_LOG_DEBUG(...) (void) 0
#endif
