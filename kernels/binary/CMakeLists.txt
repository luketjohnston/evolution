cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(binary LANGUAGES CXX CUDA)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} "/home/luke/GA/kernels/binary/tclap-1.2.5/include" )


# fetch latest argparse
#include(FetchContent)
#FetchContent_Declare(
#    argparse
#    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
#)
#FetchContent_MakeAvailable(argparse)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



add_executable(binary binary.cu "/home/luke/GA/kernels/binary/tclap-1.2.5/include/tclap/CmdLine.h")

#target_link_libraries(binary argparse "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")
target_link_libraries(binary "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")
#target_link_libraries(binary "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")

set_property(TARGET binary PROPERTY CXX_STANDARD 17)


