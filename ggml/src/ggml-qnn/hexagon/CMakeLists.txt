# GgmlOpPackage - Consolidated target for build, clean, and rebuild operations
add_custom_target(GgmlOpPackage ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning previous build artifacts..."
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_SOURCE_DIR} clean

    COMMAND ${CMAKE_COMMAND} -E echo "Building GgmlOpPackage..."
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_SOURCE_DIR}

    COMMENT "Building GgmlOpPackage (clean + build)"
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set(GGML_OP_PACKAGE_OUTPUTS
    ${CMAKE_CURRENT_SOURCE_DIR}/libhexagon.so
)

add_custom_command(
    OUTPUT ${GGML_OP_PACKAGE_OUTPUTS}
    DEPENDS GgmlOpPackage
    COMMENT "GgmlOpPackage outputs"
)

add_custom_target(GgmlOpPackageDeps DEPENDS ${GGML_OP_PACKAGE_OUTPUTS})
