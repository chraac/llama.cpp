enable_language(ASM)
cmake_policy(SET CMP0115 OLD)

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

# Base Include dirs for the Project
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/incs/HAP/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)

include_directories(${common_incs})

if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux|Windows")
    # Hexagon npu backend
    file(GLOB host_srcs "${CMAKE_CURRENT_LIST_DIR}/host/*.cpp")
    add_library(hexagon_npu_host STATIC ${host_srcs})

    build_idl(idl/hexagon_npu.idl hexagon_npu_host)

    # Find the DSP from provided domain value, default is CDSP
    choose_dsprpc(${DSP_TYPE} dsprpc)

    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set_target_properties(hexagon_npu_host PROPERTIES OUTPUT_NAME "hexagon_npu")
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux")
        target_link_options(hexagon_npu_host PUBLIC -pie)
    endif()

    link_options(hexagon_npu_host)

    # Add compile definitions to the target
    target_compile_definitions(hexagon_npu_host PUBLIC VERIFY_PRINT_ERROR)

    # Link target with custom libraries
    link_custom_library(hexagon_npu_host rpcmem)
    link_custom_library(hexagon_npu_host ${dsprpc})
else()
    cmake_minimum_required(VERSION 3.14.3)
    project(hexagon_npu C CXX ASM)

    # check if QNN_SDK_ROOT is set
    if(NOT DEFINED ENV{QNN_SDK_ROOT})
        message(FATAL_ERROR "QNN_SDK_ROOT not defined")
    endif()

    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
    message("QNN_SDK_ROOT: ${QNN_SDK_ROOT}")
    include_directories(${QNN_SDK_ROOT}/include/QNN/)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
    file(GLOB skel_srcs "${CMAKE_CURRENT_LIST_DIR}/device/*.cpp")
    add_library(hexagon_npu_skel_OBJS OBJECT ${common_srcs} ${skel_srcs})

    add_library(hexagon_npu_skel SHARED $<TARGET_OBJECTS:hexagon_npu_skel_OBJS>)

    build_idl(idl/hexagon_npu.idl hexagon_npu_skel_OBJS)

    set_target_properties(hexagon_npu_skel PROPERTIES OUTPUT_NAME "hexagon_npu_skel_${HEXAGON_ARCH}")

    copy_binaries(hexagon_npu_skel)
endif()

# vim: set noet fenc=utf-8 ff=unix ft=cmake :
