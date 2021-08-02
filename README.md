# Scene-Layers

This program implements an interactive interface to explore a scene represented by a point cloud. The objects are arranged in visibility layers, and the user can browse through them. The surface is approximated, and the occlusion relations are determined, using the DDS (discrete depth structure) [1], which is built on the same concepts of the TLDI [2]. In short, the DDS extends the cells of a view aligned 2D grid to depth piles. Points are mapped to the piles of the corresponding cells, and sorted by depth.

The input scene should be pre=segmented, such that each point has an additional field storing the object ID. Also, the scene is static at the moment. Navigation features will be added soon.

The model used is taken from the dataset available in [3]

The program is cross platform (Windows and Linux).

Note: make sure to update the compute capability and shader model versions in CMakeLists.txt:

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50 -lineinfo -cudart=static -Xptxas -v")

********************************************************************************

## Running on Windows

coming soon.

********************************************************************************

## Running on Linux

1- You will need to define the environment variables: 
* CUDA_INCLUDE_DIRECTORY, 
* CUDA_LIBRARY_DIRECTORY,
* CUDACXX. 

2- and install libpcl-dev and libcgal-dev
  
3- Then move to the directory of the project, and:

* mkdir build
* cd build
* cmake ..
* make
* ./layers


The project was tested on a Linux system, with the following specifications:

* Ubuntu-20.04
* Cuda 11.3
* gcc-9.3, g++-9.3
* CMake 3.16.3
* GeForce RTX 2070 SUPER


and also on WSL2 (Windows Subsystem for Linux 2), with the following specifications:

* Ubuntu-20.04
* Cuda 11.0
* gcc-9, g++-9
* CMake 3.16.3
* GeForce GTX 950M
* VcXsrv


[1] Fast occlusion-based point cloud exploration  (to appear in The Visual Computer Journal September 2021)

[2] Efficient collision detection while rendering dynamic point clouds (Proceedings of the 2014 Graphics Interface Conference, pages 25-33. May 2014.)
https://link.springer.com/article/10.1007/s00371-021-02243-x

[3] http://buildingparser.stanford.edu/dataset.html
