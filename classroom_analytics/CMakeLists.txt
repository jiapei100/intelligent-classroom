# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.5)

set (TARGET_NAME "classroom_analytics")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Fancy colorized output messages
string(ASCII 27 Esc)
set(CR "${Esc}[m")
set(Red "${Esc}[1;31m")
set(Green "${Esc}[1;32m")
set(Blue "${Esc}[1;34m")

# Allow users to manually set OpenCV path if desired
set(OpenCV_DIR CACHE PATH "Path to Intel OpenCV library")

set(REQUIRED_OS_ID "Ubuntu")
set(REQUIRED_OS_VERSION "16.04")
#include(osdetect)

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

set(sources include/cnn.hpp
            include/action_detector.hpp
            include/detector.hpp
            include/face_reid.hpp
            include/tracker.hpp
            include/image_grabber.hpp
            include/logger.hpp
            src/cnn.cpp
            src/action_detector.cpp
            src/detector.cpp
            src/tracker.cpp
            src/reid_gallery.cpp
            src/logger.cpp
            src/image_grabber.cpp
            src/align_transform.cpp
            main.cpp)


add_executable(${TARGET_NAME} ${sources})
link_directories(${LIB_FOLDER})


add_dependencies(${TARGET_NAME} gflags)
set_target_properties(${TARGET_NAME} ${TRAINER} PROPERTIES COMPILE_FLAGS "-pthread -std=c++11")
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
  COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} gflags)
target_link_libraries(${TARGET_NAME} /opt/intel/openvino/opencv/lib/libopencv_dnn.so)
# Install
install(TARGETS ${TARGET_NAME} DESTINATION bin)
