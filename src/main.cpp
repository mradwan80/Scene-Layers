#include "main.h"
#include "Shader.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>

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

	vector<pointCoords> Coords;
	vector<pointColor> Colors;
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
		
		//Coords.push_back(pointCoords(x,y,z));
		pointCoords p; p.x=x;p.y=y;p.z=z;
		Coords.push_back(p);

		//Colors.push_back(pointColor(r,g,b));
		pointColor c;c.r=r;c.g=g;c.b=b;
		Colors.push_back(c);
		
		ObjectIds.push_back(int(o));

		if(x<minx)	minx=x;
		if(y<miny)	miny=y;
		if(z<minz)	minz=z;
		if(x>maxx)	maxx=x;
		if(y>maxy)	maxy=y;
		if(z>maxz)	maxz=z;
	}
	int pnum=Coords.size();	//points number//
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
	glBufferData(GL_ARRAY_BUFFER, pnum * 3 * sizeof(GLfloat), &Coords[0], GL_DYNAMIC_DRAW_ARB);
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

	

	////////////
	//DDS//
	////////////
	/*cout << "start DDS\n";
	
	DDS* dds = new DDS(AVG, true, GlobalW, GlobalH, ViewWidth, &coords, &rads, &oid, ProjectionMat, vmMat, pvmMat, pvmMat);
	dds->BuildDDS();
	cout << "DDS finished\n";
	auto graphstart = now();
	OcclusionGraph::BuildGraph(dds);
	auto graphend = now();

	//OcclusionGraph::TraversOcclusions();

	dds->FreeMemory(true);
	
	dds->OutputTimings(); 
	
	cout << "graph constructed in " << graphend - graphstart << "\n";


	vector<bool>IsGraphNode(objectsnum, false);
	int graphNodes = 0;
	int graphEdges = 0;
	for (int i = 0; i < objectsnum; i++)
	{
		graphEdges += OcclusionGraph::Occludees[i].size();
		for (OcclusionStruct s : OcclusionGraph::Occludees[i])
		{
			IsGraphNode[i] = true;
			IsGraphNode[s.object] = true;
		}
			
	}
	
	for (int i = 0; i < objectsnum; i++)
	{
		if (IsGraphNode[i])
			graphNodes++;
	}
	cout << "nodes: " << graphNodes << " , edges: " << graphEdges << "\n";
	cout << "---------------------------\n";
	*/
	

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