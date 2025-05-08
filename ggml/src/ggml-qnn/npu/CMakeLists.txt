enable_language(ASM)
cmake_policy(SET CMP0115 OLD)

if(DEFINED ENV{HEXAGON_SDK_ROOT})
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
    message("HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}")
else()
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not defined")
endif()

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

# Base Include dirs for the Project
set(common_incs
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/incs/HAP/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)

include_directories(${common_incs})

function(add_device_target target_name DSP_ARCH IS_SIMULATOR)
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug|Dbg")
        set(HEXAGON_BUILD_CONFIG "Debug")
        set(EXTRA_BUILD_FLAGS
            VERBOSE=1
            TREE=1
        )
    else()
        set(HEXAGON_BUILD_CONFIG "Release")
        set(EXTRA_BUILD_FLAGS)
    endif()

    set(BUILD_CPU_COUNT 16)
    set(HEXAGON_TOOLS_VARIANT $ENV{DEFAULT_TOOLS_VARIANT})
    set(BUILD_BINARY_NAME ${CMAKE_CURRENT_LIST_DIR}/hexagon_${HEXAGON_BUILD_CONFIG}_${HEXAGON_TOOLS_VARIANT}_${DSP_ARCH}/libhexagon_npu_skel_${DSP_ARCH}.so)

    if(IS_SIMULATOR)
        set(HEXAGON_TOOLCHAIN_TYPE "hexagonsim")
        set(OUTPUT_BINARY_NAME libhexagon_npu_skel_${DSP_ARCH}_sim.so)
    else()
        set(HEXAGON_TOOLCHAIN_TYPE "hexagon")
        set(OUTPUT_BINARY_NAME libhexagon_npu_skel_${DSP_ARCH}.so)
    endif()

    add_custom_target(${target_name} ALL
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_LIST_DIR}/hexagon_${HEXAGON_BUILD_CONFIG}_${HEXAGON_TOOLS_VARIANT}_${DSP_ARCH}
        COMMAND build_cmake ${HEXAGON_TOOLCHAIN_TYPE} DSP_ARCH=${DSP_ARCH} BUILD=${CMAKE_BUILD_TYPE} ${EXTRA_BUILD_FLAGS} -j${BUILD_CPU_COUNT}
        COMMAND ${CMAKE_COMMAND} -E copy ${BUILD_BINARY_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${OUTPUT_BINARY_NAME}
        BYPRODUCTS ${BUILD_BINARY_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )
endfunction()

if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux|Windows")
    # host build
    file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
    file(GLOB host_srcs "${CMAKE_CURRENT_LIST_DIR}/host/*.cpp")
    set(stub_srcs "${CMAKE_CURRENT_BINARY_DIR}/npu_device_stub.c")
    add_library(hexagon-npu-host STATIC
        ${common_srcs}
        ${host_srcs}
        ${stub_srcs}
    )

    # disable warnings for the stub
    set_source_files_properties(
        ${stub_srcs}
        PROPERTIES
        COMPILE_FLAGS "-w"
    )

    build_idl(idl/hexagon_npu.idl hexagon-npu-host)

    # Add compile definitions to the target
    target_compile_definitions(hexagon-npu-host PUBLIC
        VERIFY_PRINT_ERROR
        GGML_QNN_ENABLE_HEXAGON_BACKEND
    )

    target_include_directories(hexagon-npu-host PRIVATE
        ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
        ${QNN_SDK_ROOT}/include/QNN/
        ${CMAKE_CURRENT_LIST_DIR}/host/
        ${CMAKE_CURRENT_LIST_DIR}/
    )

    target_include_directories(hexagon-npu-host PUBLIC
        ${HEXAGON_SDK_ROOT}/incs/ # TODO: this is for rpc-mem
    )

    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set_target_properties(hexagon-npu-host PROPERTIES OUTPUT_NAME "hexagon_npu")
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux")
        target_link_options(hexagon-npu-host PUBLIC -pie)
    endif()

    link_options(hexagon-npu-host)

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        set(PREBUILT_LIB_DIR "android_aarch64")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set(PREBUILT_LIB_DIR "UbuntuARM_aarch64")
    else()
        # Windows
        set(PREBUILT_LIB_DIR "windows_aarch64")
    endif()

    choose_dsprpc("3" dsprpc) # cdsprpc
    link_custom_library(hexagon-npu-host ${dsprpc})

    add_device_target(hexagon-npu-device-v73 "v73" FALSE)
    add_device_target(hexagon-npu-device-v75 "v75" FALSE)
    add_dependencies(hexagon-npu-host hexagon-npu-device-v73)
    add_dependencies(hexagon-npu-host hexagon-npu-device-v75)

    list(APPEND NPU_RUNTIME_LIBS "${HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonApp")
    list(APPEND NPU_RUNTIME_LIBS "${HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonAppLE")

    foreach(RUNTIME_LIB ${NPU_RUNTIME_LIBS})
        message("Copy: ${RUNTIME_LIB} -> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        add_custom_command(
            TARGET hexagon-npu-host POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${RUNTIME_LIB}
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endforeach()
else()
    # hexagon npu build
    cmake_minimum_required(VERSION 3.14.3)
    project(hexagon_npu C CXX ASM)

    # check if QNN_SDK_ROOT is set
    if(NOT DEFINED ENV{QNN_SDK_ROOT})
        message(FATAL_ERROR "QNN_SDK_ROOT not defined")
    endif()

    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
    message("QNN_SDK_ROOT: ${QNN_SDK_ROOT}")
    include_directories(
        ${QNN_SDK_ROOT}/include/QNN/
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

    file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
    file(GLOB device_srcs "${CMAKE_CURRENT_LIST_DIR}/device/*.cpp")
    set(skel_srcs "${CMAKE_CURRENT_BINARY_DIR}/npu_device_skel.c")
    add_library(hexagon_npu_skel_OBJS OBJECT
        ${common_srcs}
        ${device_srcs}
        ${skel_srcs}
    )

    if(CMAKE_BUILD_TYPE MATCHES "Debug|Dbg")
        message("Debug build, enable all logging")
        target_compile_definitions(hexagon_npu_skel_OBJS PUBLIC
            _DEBUG
            DEBUG_LOGGING
        )
    else()
        message("Release build, disable debug logging")
        target_compile_definitions(hexagon_npu_skel_OBJS PUBLIC
            NDEBUG
            RELEASE_LOGGING
        )
    endif()

    build_idl(idl/hexagon_npu.idl hexagon_npu_skel_OBJS)

    # disable warnings for the skel
    set_source_files_properties(
        ${skel_srcs}
        PROPERTIES
        COMPILE_FLAGS "-w"
    )

    add_library(hexagon_npu_skel SHARED $<TARGET_OBJECTS:hexagon_npu_skel_OBJS>)

    target_link_libraries(hexagon_npu_skel
        ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.a
        ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.a
    )
    set_target_properties(hexagon_npu_skel PROPERTIES OUTPUT_NAME "hexagon_npu_skel_${HEXAGON_ARCH}")

    copy_binaries(hexagon_npu_skel)
endif()

# vim: set noet fenc=utf-8 ff=unix ft=cmake :
