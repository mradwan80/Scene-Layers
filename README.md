# Scene-Layers

This program implements an interactive interface to explore a scene represented by a point cloud, based on revealing occluded objects. The surface is approximated, and the occlusion relations are determined, using the DDS (discrete depth structure) [1], which is built on the same concepts of the TLDI [2].

In short, the DDS extends the cells of a view aligned 2D grid to depth piles. Points are mapped to the piles of the corresponding cells, and sorted by depth. A link to a journal paper describing the structure will be provided soon.


The program is cross platform (Windows and Linux).

********************************************************************************

## Running on Windows

coming soon.

********************************************************************************

## Running on Linux

You will need to define the environment variables: 
* CUDA_INCLUDE_DIRECTORY, 
* CUDA_LIBRARY_DIRECTORY,
* CUDACXX. 
  
Then move to the directory of the project, and:

* mkdir build
* cd build
* cmake ..
* make
* ./knn


The project was tested on WSL2 (Windows Subsystem for Linux 2), with the following specifications:

* Ubuntu-20.04
* Cuda 11.0
* gcc-9, g++-9
* CMake 3.16.3
* GeForce GTX 950M
* VcXsrv

[1] Fast occlusion-based point cloud exploration  (to appear in The Visual Computer 2021)
[2] Efficient collision detection while rendering dynamic point clouds (Proceedings of the 2014 Graphics Interface Conference, pages 25-33. May 2014.)
