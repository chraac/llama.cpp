message(STATUS "Using QNN backend")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(LOG_LIB log)
    set(QNN_LINK_LIBRARIES ${LOG_LIB})
    set(QNN_DEFAULT_LIB_SEARCH_PATH "/data/local/tmp/" CACHE STRING "customized library search path for QNN backend")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "" CACHE STRING "customized library search path for QNN backend")
else()
    message(FATAL_ERROR "QNN now only available on Android, Windows and Linux")
endif()

if(NOT DEFINED GGML_QNN_SDK_PATH)
    # try read from environment variable
    # TODO: create a function to search for the SDK path
    if(DEFINED ENV{QNN_SDK_PATH})
        set(GGML_QNN_SDK_PATH $ENV{QNN_SDK_PATH})
    else()
        message(FATAL_ERROR "GGML_QNN_SDK_PATH not defined")
    endif()
endif()

set(QNN_DYNAMIC_LIBS "")

if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        # Android
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/aarch64-android")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        # Linux x86_64
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/x86_64-linux-clang")
    else()
        # Linux aarch64
        set(QNN_SDK_LIB_PATH "${GGML_QNN_SDK_PATH}/lib/aarch64-oe-linux-gcc11.2")
    endif()

    list(APPEND QNN_DYNAMIC_LIBS "${QNN_SDK_LIB_PATH}/libQnnSystem.so")
    list(APPEND QNN_DYNAMIC_LIBS "${QNN_SDK_LIB_PATH}/libQnnCpu.so")
    list(APPEND QNN_DYNAMIC_LIBS "${QNN_SDK_LIB_PATH}/libQnnGpu.so")
    list(APPEND QNN_DYNAMIC_LIBS "${QNN_SDK_LIB_PATH}/libQnnHtp.so")
    file(GLOB HTP_STUB_LIBS "${QNN_SDK_LIB_PATH}/libQnnHtp*.so")
    list(APPEND QNN_DYNAMIC_LIBS ${HTP_STUB_LIBS})

    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        file(GLOB HTP_SKEL_LIBS "${QNN_SDK_LIB_PATH}/lib/hexagon-*/unsigned/libQnnHtp*Skel.so")
        list(APPEND QNN_DYNAMIC_LIBS ${HTP_SKEL_LIBS})

        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            list(APPEND QNN_DYNAMIC_LIBS "${CMAKE_ANDROID_NDK}/prebuilt/android-arm64/gdbserver/gdbserver")
            list(APPEND QNN_DYNAMIC_LIBS "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.9/lib/linux/aarch64/libomp.so")
        endif()
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # TODO: add windows support
endif()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("QNN_SDK_PATH: ${GGML_QNN_SDK_PATH}")

file(GLOB QNN_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
ggml_add_backend_library(ggml-qnn
    ${QNN_SOURCES}
)

target_include_directories(ggml-qnn PRIVATE ${GGML_QNN_SDK_PATH}/include/QNN ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(ggml-qnn PRIVATE ${QNN_LINK_LIBRARIES})

if(NOT "${QNN_DEFAULT_LIB_SEARCH_PATH}" STREQUAL "")
    string(REGEX REPLACE "/$" "" QNN_DEFAULT_LIB_SEARCH_PATH "${QNN_DEFAULT_LIB_SEARCH_PATH}")
endif()

message("GGML_QNN_DEFAULT_LIB_SEARCH_PATH: ${QNN_DEFAULT_LIB_SEARCH_PATH}")
target_compile_definitions(ggml-qnn PRIVATE GGML_QNN_DEFAULT_LIB_SEARCH_PATH="${QNN_DEFAULT_LIB_SEARCH_PATH}")

if(GGML_QNN_ENABLE_CPU_BACKEND)
    message("GGML_QNN_ENABLE_CPU_BACKEND is enabled")
    target_compile_definitions(ggml-qnn PRIVATE GGML_QNN_ENABLE_CPU_BACKEND)
else()
    message("GGML_QNN_ENABLE_CPU_BACKEND is disabled")
endif()

if(GGML_QNN_ENABLE_PERFORMANCE_TRACKING)
    message("GGML_QNN_ENABLE_PERFORMANCE_TRACKING is enabled")
    target_compile_definitions(ggml-qnn PRIVATE GGML_QNN_ENABLE_PERFORMANCE_TRACKING)
else()
    message("GGML_QNN_ENABLE_PERFORMANCE_TRACKING is disabled")
endif()

# foreach QNN_DYNAMIC_LIBS
foreach(QNN_DYNAMIC_LIB ${QNN_DYNAMIC_LIBS})
    message("Copy: ${QNN_DYNAMIC_LIB} -> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    add_custom_command(
        TARGET ggml-qnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${QNN_DYNAMIC_LIB}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()
