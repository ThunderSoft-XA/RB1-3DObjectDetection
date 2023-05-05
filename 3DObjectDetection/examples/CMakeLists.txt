#Cmake lowest  version 
CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
 
#project name
PROJECT(demo_test CXX)
 
#allow gdb debug
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
#add C++11 support and other options
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs -Wno-deprecated")
#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS -g -ftest-coverage -fprofile-arcs"})

add_definitions(-w)

#set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "    OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST            REQUIRED gstreamer-1.0)
#pkg_check_modules(GSTRTSPSERVER REQUIRED gstreamer-rtsp-server-1.0)
#pkg_check_modules(GSTRTSP       REQUIRED gstreamer-rtsp-1.0)
pkg_check_modules(GSTAPP         REQUIRED gstreamer-app-1.0)
pkg_check_modules(GLIB           REQUIRED glib-2.0)
pkg_check_modules(GFLAGS         REQUIRED gflags)

find_package(Boost)
IF (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    add_definitions( "-DHAS_BOOST" )
ENDIF()

message(STATUS "OpenCV:${OpenCV_INCLUDE_DIRS},${OpenCV_LIBRARY_DIRS},${OpenCV_LIBRARIES}")
message(STATUS "GST:   ${GST_INCLUDE_DIRS},${GST_LIBRARY_DIRS},${GST_LIBRARIES}")
message(STATUS "GSTAPP:${GSTAPP_INCLUDE_DIRS},${GSTAPP_LIBRARY_DIRS},${GSTAPP_LIBRARIES}")
message(STATUS "GLIB:  ${GLIB_INCLUDE_DIRS},${GLIB_LIBRARY_DIRS},${GLIB_LIBRARIES}")
message(STATUS "GFLAGS:${GFLAGS_INCLUDE_DIRS},${GFLAGS_LIBRARY_DIRS},${GFLAGS_LIBRARIES}")

message(INFO "source file: " ${DIR_SRCS})

#Tensorflow
set(TFLITE_LIBS tflite)
IF (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    MESSAGE(STATUS "current hardware platform: x86_64 ")
    set(TENSORFLOW_LIBRARY_DIRS "/home/thundersoft/software/tensorflow/bazel-bin/tensorflow/lite")
ELSEIF (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    MESSAGE(STATUS "current hardware platform: aarch64")
    set(TENSORFLOW_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../inference/libs/")
ENDIF ()

# include file dir
include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/include/glib-2.0/
    ${BOOST_INCLUDE_DIR}
    ${GST_INCLUDE_DIRS}
    ${GSTAPP_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}	
    ${GFLAGS_INCLUDE_DIRS}
    $ENV{TENSORFLOW_ROOT}/
    $ENV{TENSORFLOW_THIRD_PARTY}/
    ../gst_component
    ../inference
    ../opencv_app
    ../jsoncpp/include
)

add_executable( gst_3D_object_test
    ${CMAKE_CURRENT_SOURCE_DIR}/gst_3D_object.cpp
)

# #add link libs
target_link_libraries( gst_3D_object_test
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    inference
    jsoncpp_lib
    pthread
    glib-2.0
    m
    z
)
