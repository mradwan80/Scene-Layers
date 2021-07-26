#include "main.h"
#include "Shader.h"
#include "DDS.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>

#include "OcclusionGraph.h"
#include "Browser.h"
#include "VisibilityUpdater.h"


Browser* browser;
OcclusionGraph* graph;
VisibilityUpdater* vadapter;

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	
	if (action==GLFW_RELEASE)
		return;

	switch (key)
	{
		
		case GLFW_KEY_ESCAPE:
			if (action == GLFW_PRESS)
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			break;
		case GLFW_KEY_UP:
			if(browser->incrementLayer())
				vadapter->UpdateVisibility(browser);
			break;

		case GLFW_KEY_DOWN:
			if(browser->decrementLayer())
				vadapter->UpdateVisibility(browser);
			break;
	}
	
}

int main()
{
	int GlobalW=1024;
	int GlobalH=800;

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
	vector<float>Rads;	//constant for now//
	vector<int>ObjectIds;
	vector<float>Visible;
	
	//read data from file//
	//std::ifstream inputfile("../models/three_bunnies.xyz", std::ios_base::in);
	std::ifstream inputfile("../models/conference-room.xyz", std::ios_base::in);
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
		pointColor c;c.r=r/255;c.g=g/255;c.b=b/255;
		Colors.push_back(c);
		
		ObjectIds.push_back(int(o));

		Rads.push_back(0.05);


		Visible.push_back(1);

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
	float ViewWidth=(maxx-minx)*1.5;

	//find number of objects//
	std::vector<int>::iterator maxobject = std::max_element(ObjectIds.begin(), ObjectIds.end());
	int objectsnum=maxobject[0]+1; // assuming all IDs between 0 and maxobject are used//

	//vao and vbos//
	GLuint vao;
	vector<GLuint> vbo;
	vbo.resize(4);
	glGenVertexArrays(1, &vao); 
	glGenBuffers(4, &vbo[0]); 
	glBindVertexArray(vao); 
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, pnum * 3 * sizeof(GLfloat), &Coords[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, pnum * 3 * sizeof(GLfloat), &Colors[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, pnum * 1 * sizeof(GLfloat), &Visible[0], GL_DYNAMIC_DRAW_ARB);
	//glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
	//glBufferData(GL_ARRAY_BUFFER, pnum * 1 * sizeof(GLint), &ObjectIds[0], GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	vadapter = VisibilityUpdater::getInstance(pnum, vao, vbo[2], &ObjectIds);

	//matrices//
	glm::mat4 ModelMat = glm::translate(glm::vec3(0,0,0));
	glm::mat4 ViewMat = glm::translate(glm::vec3(-midx,-midy,-(minz+1.95*(maxz-midz)))); //for conference room
	//glm::mat4 ViewMat = glm::translate(glm::vec3(-midx,-midy,-(minz+3*(maxz-midz))));	//for 3 bunnies
	float fov=70.0;
	float Near = 0.01;
	float Far = 7*(maxz-minz);
	glm::mat4 ProjectionMat =	glm::perspective(fov, float(GlobalW)/float(GlobalH) , Near, Far);
	glm::mat4 vmMat = ViewMat*ModelMat;
	glm::mat4 pvmMat = ProjectionMat*ViewMat*ModelMat;


	//read shader//
	Shader PntRdr("../shaders/PointsRenderer");
	PntRdr.CompileShader();
	

	glEnable(GL_PROGRAM_POINT_SIZE);
	glViewport(0, 0, GlobalW, GlobalH);
	glEnable(GL_DEPTH_TEST);

	

	////////////
	//DDS//
	////////////
	cout << "start DDS\n";
	
	DDS* dds = new DDS(AVG, true, GlobalW, GlobalH, ViewWidth, &Coords, &Rads, &ObjectIds, ProjectionMat, vmMat, pvmMat, pvmMat);
	dds->BuildDDS();
	cout << "DDS finished\n";

	////////////
	////////////
	////////////

	graph = OcclusionGraph::getInstance();

	graph->BuildGraph(dds, objectsnum);

	dds->FreeMemory(true);
	
	cout << "---------------------------\n";
	
	graph->TraverseOcclusions();

	glfwSetKeyCallback(myWindow, key_callback);

	browser = Browser::getInstance();
	browser->CreateLayers(objectsnum);

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
		glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
		glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0); 
		glEnableVertexAttribArray(2);
		glUniformMatrix4fv(glGetUniformLocation(PntRdr.GetHandle(), "pvm_matrix"), 1, GL_FALSE, glm::value_ptr(pvmMat));
		glDrawArrays(GL_POINTS, 0, pnum);
		

		glfwSwapBuffers(myWindow);
		glfwPollEvents(); 
	}


	return 1;
}
