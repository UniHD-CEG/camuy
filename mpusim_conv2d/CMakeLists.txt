# Copyright 2019, 2020 Kevin Stehle
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.10)

include(DefaultFlags.cmake)

set(MPUSIM_CONV2D_EIGEN3_INSTALL_DIR "" CACHE STRING "Install directory of eigen3, required if installed locally")
option(MPUSIM_CONV2D_EIGEN3_LOCAL_INSTALL "Use local installation of eigen3" OFF)

if(NOT MPUSIM_CONV2D_EIGEN3_LOCAL_INSTALL)
    find_package(Eigen3 REQUIRED)
else()
    find_package(Eigen3 REQUIRED HINTS "${MPUSIM_CONV2D_EIGEN3_INSTALL_DIR}/share/eigen3")
endif()

set(MPUSIM_CONV2D_MPUSIM_INCLUDE_DIR "" CACHE STRING "Directory of mpusim header files")
set(MPUSIM_CONV2D_MPUSIM_WRAPPER_INCLUDE_DIR "" CACHE STRING "Directory of mpusim wrapper header files")
set(MPUSIM_CONV2D_MPUSIM_WRAPPER_INSTALL_DIR "" CACHE STRING "Directory of libmpusim-wrapper.so")
set(MPUSIM_CONV2D_TENSORFLOW_INSTALL_DIR "" CACHE STRING "Directory of tensorflow libraries")

#General

project(mpusim-conv2d)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I../tensorflow -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ltensorflow_framework")

include_directories("${MPUSIM_CONV2D_EIGEN3_INSTALL_DIR}/include" ${MPUSIM_CONV2D_MPUSIM_INCLUDE_DIR} ${MPUSIM_CONV2D_MPUSIM_WRAPPER_INCLUDE_DIR})

link_directories(${MPUSIM_CONV2D_TENSORFLOW_INSTALL_DIR} ${MPUSIM_CONV2D_MPUSIM_WRAPPER_INSTALL_DIR})

add_library(${PROJECT_NAME} SHARED mpusim_conv2d.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS OFF)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE "libmpusim-wrapper.so")
