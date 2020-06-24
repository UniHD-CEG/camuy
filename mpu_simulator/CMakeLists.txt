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

cmake_minimum_required(VERSION 3.8)

include(DefaultFlags.cmake)

set(MPUSIM_EIGEN3_INSTALL_DIR "" CACHE STRING "Install directory of eigen3, required if installed locally")
option(MPUSIM_EIGEN3_LOCAL_INSTALL "Use local installation of eigen3" OFF)

project(mpusim)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -Wpedantic -pedantic-errors \
-fasynchronous-unwind-tables -fexceptions \
-fstack-protector-strong -Wl,-z,defs -Wl,-z,now -Wl,-z,relro -grecord-gcc-switches -fopenmp")

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")

if(NOT MPUSIM_EIGEN3_LOCAL_INSTALL)
    find_package(Eigen3 REQUIRED)
else()
    find_package(Eigen3 REQUIRED HINTS "${MPUSIM_EIGEN3_INSTALL_DIR}/share/eigen3")
    include_directories("${MPUSIM_EIGEN3_INSTALL_DIR}/include")
endif()

include_directories(include/)

link_directories(${CMAKE_BINARY_DIR})

#mpusim

set(MPUSIM_SOURCES include/mpu_exception.h
                            include/processing_element.h
                            include/processing_element_top_border.h
                            include/processing_element_left_border.h
                            include/processing_element_center.h
                            include/activation_fifo.h
                            include/systolic_array.h
                            include/systolic_data_setup_unit.h
                            include/weight_fetcher.h
                            include/accumulator_array.h
                            include/memory_management_unit.h
                            include/matrix_processing_unit.h
                            include/mpu_statistics_log_entry.h
                            include/mpu_statistics_logger.h)

add_library(${PROJECT_NAME} SHARED ${MPUSIM_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS OFF)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER include/matrix_processing_unit.h)

#mpusim_test

add_executable(mpusim_test "test/mpu_simulator_test.cpp")
set_target_properties(mpusim_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(mpusim_test PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS OFF)
target_link_libraries(mpusim_test PRIVATE ${PROJECT_NAME})
target_link_libraries(mpusim_test PRIVATE Eigen3::Eigen)


