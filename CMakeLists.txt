cmake_minimum_required(VERSION 3.4.1)
project(nndescent)

set(CMAKE_CXX_STANDARD 11)
find_package(Python3 COMPONENTS Development NumPy)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}    
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    #${CMAKE_CURRENT_SOURCE_DIR}/../matplotlib-cpp/
)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

add_executable(nndescent main.cpp)
target_link_libraries(nndescent Python3::Python Python3::NumPy)