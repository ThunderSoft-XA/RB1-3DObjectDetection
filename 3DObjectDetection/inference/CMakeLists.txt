# The inference examples example for Tensorflow Lite.

cmake_minimum_required(VERSION 3.4.1)

PROJECT(inference_tf CXX)

# base Cmake config
# allow gdb debug
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
 
#add C++11 support and other options
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs -Wno-deprecated")
#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS -g -ftest-coverage -fprofile-arcs"})

set(CODE_ROOT /home/HDD/codes/rb2/LE.UM.5.4.1/apps_proc/build-qti-distro-fullstack-debug/tmp-glibc/work)
set(TF_SOURCE_DIR ${CODE_ROOT}/aarch64-oe-linux/tensorflow-lite/2.6.0-r0/git/tensorflow)
set(TFLITE_SOURCE_DIR ${TF_SOURCE_DIR}/lite)

message(STATUS "======================================================================")

#set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)

# add profiling and tools from tflite source code
list(APPEND TFLITE_INFERENCE_SRCS
  ${TF_SOURCE_DIR}/core/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/sparsity/format_converter.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
  ${TFLITE_SOURCE_DIR}/delegates/utils/simple_delegate.cc
)

message(STATUS "=======TFLITE_INFERENCE_SRCS====：${TFLITE_INFERENCE_SRCS}")

set(TFLITE_ENABLE_GPU 1)
set(TFLITE_ENABLE_HEXAGON 1)

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_INFERENCE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
  )
else()
  set(TFLITE_INFERENCE_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_INFERENCE_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_INFERENCE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

if(TFLITE_ENABLE_HEXAGON)
  list(APPEND TFLITE_INFERENCE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/hexagon_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_HEXAGON

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${TF_SOURCE_DIR}/../
    ${OpenCV_INCLUDE_DIRS}
)

# add_executable
add_library(inference
#   EXCLUDE_FROM_ALL
    # ${OpenCV_LIBS}
    ${TFLITE_INFERENCE_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/inference_tf.cpp
)

add_executable(object_3d
#   EXCLUDE_FROM_ALL
    # ${OpenCV_LIBS}
    ${TFLITE_INFERENCE_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/inference_tf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/3d_object_tf_test.cpp
)

install(TARGETS inference
    RUNTIME
)

target_compile_options(inference
    PRIVATE
        ${TFLITE_INFERENCE_CC_OPTIONS}
)
target_link_libraries(inference
    tensorflow-lite
)

target_link_libraries(object_3d
    ${OpenCV_LIBS}
    tensorflow-lite
)
