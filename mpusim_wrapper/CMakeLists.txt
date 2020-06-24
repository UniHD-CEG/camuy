# Copyright (c) 2020 Computing Systems Group
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

set(MPUSIM_WRAPPER_EIGEN3_INSTALL_DIR "" CACHE STRING "Install directory of eigen3, required if installed locally")
option(MPUSIM_WRAPPER_EIGEN3_LOCAL_INSTALL "Use local installation of eigen3" OFF)

if(NOT MPUSIM_WRAPPER_EIGEN3_LOCAL_INSTALL)
    find_package(Eigen3 REQUIRED)
else()
    find_package(Eigen3 REQUIRED HINTS "${MPUSIM_WRAPPER_EIGEN3_INSTALL_DIR}/share/eigen3")
endif()

set(MPUSIM_WRAPPER_MPUSIM_INCLUDE_DIR "" CACHE STRING "Directory of mpusim header files")
set(MPUSIM_WRAPPER_MPUSIM_INSTALL_DIR "" CACHE STRING "Directory of libmpusim.so")

#General

project(mpusim-wrapper)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -Wpedantic -pedantic-errors \
-fasynchronous-unwind-tables -fexceptions \
-fstack-protector-strong -Wl,-z,defs -Wl,-z,now -Wl,-z,relro -grecord-gcc-switches -fopenmp")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address")

include_directories("${MPUSIM_WRAPPER_EIGEN3_INSTALL_DIR}/include" ${MPUSIM_WRAPPER_MPUSIM_INCLUDE_DIR})

link_directories(${MPUSIM_WRAPPER_MPUSIM_INSTALL_DIR})

set(MPUSIM_WRAPPER_SOURCES mpusim_wrapper.h
                            mpusim_wrapper.cpp)

add_library(${PROJECT_NAME} SHARED ${MPUSIM_WRAPPER_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS OFF)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE "libmpusim.so")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER mpusim_wrapper.h)
