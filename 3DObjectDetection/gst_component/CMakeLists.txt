#Cmake lowest  version 
CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
 
#project name
PROJECT(gst_depth_test CXX)

#set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST            REQUIRED gstreamer-1.0)
pkg_check_modules(GSTRTSPSERVER REQUIRED gstreamer-rtsp-server-1.0)
pkg_check_modules(GSTRTSP       REQUIRED gstreamer-rtsp-1.0)
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
    ../jsoncpp/include
)
 
link_directories(
    ${GST_LIBRARY_DIRS}
    ${GSTAPP_LIBRARY_DIRS}
    ${GLIB_LIBRARY_DIRS}
    ${GFLAGS_LIBRARY_DIRS}
    ${BOOST_LIBRARYDIR}
)

# get a executabl file
add_executable(camera2appsink
    # ${CMAKE_CURRENT_SOURCE_DIR}/camera2appsink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_test.cpp
) 

# #get a executabl file
add_executable(appsrc2udpsink
    # ${CMAKE_CURRENT_SOURCE_DIR}/appsrc2udpsink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/appsrc_udp_test.cpp
)

# #get a executabl file
add_executable(appsrc2rtsp
    # ${CMAKE_CURRENT_SOURCE_DIR}/appsrc2rtsp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/appsrc_rtsp_test.cpp
)

# #get a executabl file
add_executable(cv_rtsp_test
    # ${CMAKE_CURRENT_SOURCE_DIR}/appsrc2rtsp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv_rtsp_test.cpp
)

#add link libs
target_link_libraries(camera2appsink
    ${OpenCV_LIBS}
    ${BOOST_LIBRARYDIR}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    jsoncpp_lib
    pthread
    glib-2.0
    m
    z
)

# #add link libs
target_link_libraries(appsrc2rtsp
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    jsoncpp_lib
    pthread
    glib-2.0
    m
    z
)

# #add link libs
target_link_libraries(appsrc2udpsink
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    jsoncpp_lib
    pthread
    glib-2.0
    m
    z
)

# #add link libs
target_link_libraries(cv_rtsp_test
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    jsoncpp_lib
    pthread
    glib-2.0
    m
    z
)