cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)
find_package(PythonLibs REQUIRED)
message("Source dir: ${CMAKE_SOURCE_DIR}")
include_directories(${PYTHON_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/tclap-1.2.5/include" )


# fetch latest argparse
#include(FetchContent)
#FetchContent_Declare(
#    argparse
#    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
#)
#FetchContent_MakeAvailable(argparse)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")




add_executable(main main.cu binary.cu binary.cuh)

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

#target_link_libraries(binary argparse "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")
target_link_libraries(main "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")
#target_link_libraries(binary "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 17)


