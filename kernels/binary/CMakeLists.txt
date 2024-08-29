cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(binary LANGUAGES CXX CUDA)

#add_library(binary STATIC 
#debug.h
#)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(binary binary.cu)
target_link_libraries(binary "${TORCH_LIBRARIES}" ${PYTHON_LIBRARIES})
set_property(TARGET binary PROPERTY CXX_STANDARD 17)
