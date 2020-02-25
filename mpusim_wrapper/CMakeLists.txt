cmake_minimum_required(VERSION 3.10)

include(DefaultFlags.cmake)

set(MPUSIM__EIGEN3_INSTALL_DIR "" CACHE STRING "Install directory of eigen3, required if installed locally")
option(MPUSIM_WRAPPER_EIGEN3_LOCAL_INSTALL "Use local installation of eigen3" OFF)

if(NOT MPUSIM_WRAPPER_EIGEN3_LOCAL_INSTALL)
    find_package(Eigen3 REQUIRED)
else()
    find_package(Eigen3 REQUIRED HINTS "${MPUSIM_WRAPPER_EIGEN3_INSTALL_DIR}/share/eigen3")
endif()

set(MPUSIM_WRAPPER_MPUSIM_INCLUDE_DIR "" CACHE STRING "Directory of mpusim header files")
set(MPUSIM_WRAPPER_MPUSIM_LIB_DIR "" CACHE STRING "Directory of libmpusim.so")

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

link_directories(${MPUSIM_WRAPPER_MPUSIM_LIB_DIR})

set(MPUSIM_WRAPPER_SOURCES mpusim_wrapper.h
                            mpusim_wrapper.cpp)

add_library(${PROJECT_NAME} SHARED ${MPUSIM_WRAPPER_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS OFF)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE "libmpusim.so")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER mpusim_wrapper.h)
