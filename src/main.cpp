
#include <iostream>
#include <fstream>
#include <vector>
#include "../external/glew-1.10.0/include/GL/glew.h"
#include "../external/glfw/include/GLFW/glfw3.h"
#include "../external/glm/glm.hpp"
#include "../external/glm/gtx/transform.hpp"
#include "../external/glm/gtc/type_ptr.hpp"

#include "Shader.h"

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>

void func(int size, int* a1, int* a2, int* a3);
void FillWithValue(int* arr, int size, int val);

struct Position
{
	float x,y,z;
};

struct Color
{
	float r,g,b;
};

int main()
{
	int GlobalW=800;
	int GlobalH=600;

	if (!glfwInit())
	{
		std::cout << "GLFW Initialization failed" << std::endl;
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	GLFWwindow* myWindow = glfwCreateWindow(GlobalW, GlobalH, "DDS-Layers", NULL, NULL);
	if (!myWindow)
	{
		std::cout << "failed to create GLFW Window" << std::endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(myWindow);
	glfwSwapInterval(0);


	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	vector<Position> Positions;
	vector<Color> Colors;
	vector<int>ObjectIds;

	//read data from file//
	//std::ifstream inputfile("../models/bunny.xyz", std::ios_base::in);
	std::ifstream inputfile("../models/three_bunnies.xyz", std::ios_base::in);
	float x,y,z,r,g,b,o;
	float maxx=std::numeric_limits<float>::lowest();
	float maxy=std::numeric_limits<float>::lowest();
	float maxz=std::numeric_limits<float>::lowest();
	float minx=std::numeric_limits<float>::max();
	float miny=std::numeric_limits<float>::max();
	float minz=std::numeric_limits<float>::max();
	while(!inputfile.eof())
	{
		inputfile>>x>>y>>z>>r>>g>>b>>o;
		
		//Positions.push_back(Position(x,y,z));
		Position p; p.x=x;p.y=y;p.z=z;
		Positions.push_back(p);

		//Colors.push_back(Color(r,g,b));
		Color c;c.r=r;c.g=g;c.b=b;
		Colors.push_back(c);
		
		ObjectIds.push_back(int(o));

		if(x<minx)	minx=x;
		if(y<miny)	miny=y;
		if(z<minz)	minz=z;
		if(x>maxx)	maxx=x;
		if(y>maxy)	maxy=y;
		if(z>maxz)	maxz=z;
	}
	int pnum=Positions.size();	//points number//
	float midx=(minx+maxx)/2;
	float midy=(miny+maxy)/2;
	float midz=(minz+maxz)/2;
	

	//vao and vbos//
	GLuint vao;
	vector<GLuint> vbo(3);
	glGenVertexArrays(1, &vao); 
	glGenBuffers(3, &vbo[0]); 
	glBindVertexArray(vao); 
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, pnum * 3 * sizeof(GLfloat), &Positions[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, pnum * 3 * sizeof(GLfloat), &Colors[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, pnum * 1 * sizeof(GLint), &ObjectIds[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//matrices//
	glm::mat4 ModelMat = glm::translate(glm::vec3(0,0,0));
	glm::mat4 ViewMat = glm::translate(glm::vec3(-midx,-midy,-(minz+2.5*(maxz-midz))));
	float fov=70.0;
	float Near = 0.01;
	float Far = 7*(maxz-minz);
	glm::mat4 ProjectionMat =	glm::perspective(fov, float(GlobalW)/float(GlobalH) , Near, Far);
	glm::mat4 PVMMat = ProjectionMat*ViewMat*ModelMat;


	//read shader//
	Shader PntRdr("../shaders/PointsRenderer");
	PntRdr.CompileShader();
	

	glEnable(GL_PROGRAM_POINT_SIZE);
	glViewport(0, 0, GlobalW, GlobalH);
	glEnable(GL_DEPTH_TEST);

	
	/////////
	//testing cuda
	////////
	/*
	int size=1000;
	int *arr1, *arr2, *arr3;
	
	cudaMalloc((void**)&arr1, size * sizeof(int));
	FillWithValue(arr1,size,1);

	cudaMalloc((void**)&arr2, size * sizeof(int));
	FillWithValue(arr2,size,2);

	cudaMalloc((void**)&arr3, size * sizeof(int));

	int* harr = new int [size];
	cudaMemcpy(harr,arr1,size*sizeof(int),cudaMemcpyDeviceToHost);
	fprintf(stdout, "%d\n",harr[0]);


	func(size, arr1, arr2, arr3);
	cudaError_t errc = cudaGetLastError();
	if (cudaSuccess != errc)
		fprintf(stderr, "Cuda error: %s.\n", cudaGetErrorString(errc));
	*/
	//////////
	//////////
	//////////

	//render loop//
	while (!glfwWindowShouldClose(myWindow))
	{
		glClearDepth(1.0);
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		
		glUseProgram(PntRdr.GetHandle());
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0); 
		glEnableVertexAttribArray(1);
		glUniformMatrix4fv(glGetUniformLocation(PntRdr.GetHandle(), "pvm_matrix"), 1, GL_FALSE, glm::value_ptr(PVMMat));
		glDrawArrays(GL_POINTS, 0, pnum);
		

		glfwSwapBuffers(myWindow);
		glfwPollEvents(); 
	}


	return 1;
}