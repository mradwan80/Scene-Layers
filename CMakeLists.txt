cmake_minimum_required(VERSION 2.8)
project(proj LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)

#set(CMAKE_VERBOSE_MAKEFILE ON)

#### detect operating system ####
#set(SYSTEM "Unix")
#set(SYS "${SYSTEM}${ARCH}")

#### use cuda ####
if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50 -lineinfo -cudart=static -Xptxas -v")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50 -lineinfo -cudart=static -Xptxas -v")
endif()

#	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21;-lineinfo;-Xptxas -v)
#	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo;-rdc=true; -cudart=static; -Xptxas; -v)


#### use glew ####
find_package(GLEW)
include_directories(${GLEW_INCLUDE_DIRS})

#### use glfw ####
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

include_directories($ENV{CUDA_INCLUDE_DIRECTORY})
link_directories($ENV{CUDA_LIBRARY_DIRECTORY})

ADD_EXECUTABLE(
proj src/main.cpp
src/Shader.cpp
src/cudafile.cu
)

target_link_libraries(proj ${GLFW_STATIC_LIBRARIES} GLEW GLU GL ${GLEW_LIBRARIES} )
