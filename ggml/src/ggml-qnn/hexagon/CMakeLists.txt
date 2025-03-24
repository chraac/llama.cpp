
if(NOT DEFINED GGML_HEXAGON_SDK_PATH)
    # try read from environment variable
    # TODO: create a function to search for the SDK path
    if(DEFINED ENV{HEXAGON_SDK_PATH})
        set(GGML_HEXAGON_SDK_PATH $ENV{HEXAGON_SDK_PATH})
    else()
        message(FATAL_ERROR "GGML_HEXAGON_SDK_PATH not defined")
    endif()
endif()

message("QNN_SDK_PATH: ${GGML_QNN_SDK_PATH}")
message("HEXAGON_SDK_PATH: ${GGML_HEXAGON_SDK_PATH}")

# GgmlOpPackage - Consolidated target for build, clean, and rebuild operations
add_custom_target(GgmlOpPackage ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning previous build artifacts..."
    COMMAND ${CMAKE_MAKE_PROGRAM} -C "${CMAKE_CURRENT_SOURCE_DIR}/GgmlOpPackage" clean

    COMMAND ${CMAKE_COMMAND} -E echo "Building GgmlOpPackage..."
    COMMAND ${CMAKE_MAKE_PROGRAM} -C "${CMAKE_CURRENT_SOURCE_DIR}/GgmlOpPackage"
        QNN_SDK_ROOT=${GGML_QNN_SDK_PATH}
        HEXAGON_SDK_ROOT=${GGML_HEXAGON_SDK_PATH}

    COMMENT "Building GgmlOpPackage (clean + build)"
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set(GGML_OP_PACKAGE_OUTPUTS
    ${CMAKE_CURRENT_SOURCE_DIR}/libQnnGgmlOpPackage_HTP.so
    ${CMAKE_CURRENT_SOURCE_DIR}/libQnnGgmlOpPackage_CPU.so
)

add_custom_command(
    OUTPUT ${GGML_OP_PACKAGE_OUTPUTS}
    DEPENDS GgmlOpPackage
    COMMENT "GgmlOpPackage outputs"
)

add_custom_target(GgmlOpPackageDeps DEPENDS ${GGML_OP_PACKAGE_OUTPUTS})
