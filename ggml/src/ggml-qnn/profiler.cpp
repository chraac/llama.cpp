
#include "profiler.hpp"

#include "logger.hpp"
#include "qnn-lib.hpp"

namespace qnn {

qnn_event_tracer::qnn_event_tracer(std::shared_ptr<qnn_interface> interface, Qnn_BackendHandle_t backend_handle,
                                   sdk_profile_level level) :
    _interface(interface) {
    QnnProfile_Level_t qnn_profile_level = 0;
    switch (level) {
        case sdk_profile_level::PROFILE_BASIC:
            qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
            break;
        case sdk_profile_level::PROFILE_OP_TRACE:
        case sdk_profile_level::PROFILE_DETAIL:
            qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
            break;
        case sdk_profile_level::PROFILE_OFF:
        default:
            QNN_LOG_WARN("Invalid profile level %d, using PROFILE_OFF", level);
            return;
    }

    auto error = _interface->qnn_profile_create(backend_handle, qnn_profile_level, &_handle);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("Failed to create QNN profile_handle. Backend ID %u, error %ld", _interface->get_backend_id(),
                      (long) QNN_GET_ERROR_CODE(error));
        _handle = nullptr;
        return;
    }

    if (level == sdk_profile_level::PROFILE_OP_TRACE) {
        QnnProfile_Config_t qnn_profile_config                     = QNN_PROFILE_CONFIG_INIT;
        qnn_profile_config.option                                  = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
        std::array<const QnnProfile_Config_t *, 2> profile_configs = { &qnn_profile_config, nullptr };
        error = _interface->qnn_profile_set_config(_handle, profile_configs.data());
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("Failed to set QNN profile event. Backend ID %u, error %ld", _interface->get_backend_id(),
                          (long) QNN_GET_ERROR_CODE(error));
            _interface->qnn_profile_free(_handle);
            _handle = nullptr;
            return;
        }
    }
}

qnn_event_tracer::~qnn_event_tracer() {
    if (_handle) {
        Qnn_ErrorHandle_t error = _interface->qnn_profile_free(_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("Failed to free QNN profile_handle. Backend ID %u, error %ld", _interface->get_backend_id(),
                          (long) QNN_GET_ERROR_CODE(error));
        }
        _handle = nullptr;
    }
}

void qnn_event_tracer::print_profile_events() {
    // TODO: Implement this function
}

}  // namespace qnn
