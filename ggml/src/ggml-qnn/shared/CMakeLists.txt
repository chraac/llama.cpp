
file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(runtime-common STATIC
    ${common_srcs}
)

if(DEFINED ENV{HEXAGON_SDK_ROOT})
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
    target_include_directories(runtime-common PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/
        ${CMAKE_CURRENT_LIST_DIR}/../
        ${CMAKE_CURRENT_LIST_DIR}/../../
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/ # TODO: figure out how to remove this
        ${HEXAGON_SDK_ROOT}/incs/
        ${HEXAGON_SDK_ROOT}/incs/stddef/
        ${HEXAGON_SDK_ROOT}/incs/HAP/
        ${HEXAGON_SDK_ROOT}/rtos/qurt/
        ${HEXAGON_SDK_ROOT}/utils/examples/
    )
    target_compile_definitions(runtime-common PRIVATE
        GGML_QNN_ENABLE_HEXAGON_PACKAGE
    )
    message("found HEXAGON_SDK_ROOT, appending to include directories")
else()
    message("HEXAGON_SDK_ROOT not defined, not appending to include directories")
endif()
