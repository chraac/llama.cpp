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

    # Build interface files using user IDL
    build_idl(idl/hexagon_npu.idl hexagon_npu_host)

    # Find the DSP from provided domain value, default is CDSP
    choose_dsprpc(${DSP_TYPE} dsprpc)

    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set_target_properties(hexagon_npu_host PROPERTIES OUTPUT_NAME "hexagon_npu")
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux")
        target_link_options(hexagon_npu_host PUBLIC -pie)
    endif()

    # Add common HLOS-specific linker options
    link_options(hexagon_npu_host)

    # Add compile definitions to the target
    target_compile_definitions(hexagon_npu_host PUBLIC VERIFY_PRINT_ERROR)

    # Link target with custom libraries
    link_custom_library(hexagon_npu_host rpcmem)
    link_custom_library(hexagon_npu_host getopt_custom)
    link_custom_library(hexagon_npu_host ${dsprpc})

    add_dependencies(hexagon_npu_host hap_example)
    target_link_libraries(hap_example_device hap_example)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(hexagon_npu_host hexagon_npu)

else()
    file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
    file(GLOB skel_srcs "${CMAKE_CURRENT_LIST_DIR}/device/*.cpp")
    add_library(hexagon_npu_skel_OBJS OBJECT ${common_srcs} ${skel_srcs})

    # creating shared skel library
    add_library(hexagon_npu_skel SHARED $<TARGET_OBJECTS:hexagon_npu_skel_OBJS>)

    build_idl(idl/hexagon_npu.idl hexagon_npu_skel_OBJS)

    set(q_exec_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src_app/hexagon_npu_main
        ${CMAKE_CURRENT_SOURCE_DIR}/src_app/hexagon_npu
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils
    )

    add_library(hexagon_npu_skel_static STATIC $<TARGET_OBJECTS:hexagon_npu_skel_OBJS>)
    add_library(hexagon_npu_q SHARED ${q_exec_srcs})

    # Link target with custom libraries
    link_custom_library(hexagon_npu_q rpcmem)

    add_dependencies(hexagon_npu_q hexagon_npu_skel_static)
    target_link_libraries(hexagon_npu_q hexagon_npu_skel_static)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(hexagon_npu_q hexagon_npu_skel)
endif()

# vim: set noet fenc=utf-8 ff=unix ft=cmake :
