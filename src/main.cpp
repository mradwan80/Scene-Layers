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

int MaxObject;
int objectsnum;
vector<int>OrderedIDs;
vector<int>RealIDs;

struct OcclusionStruct
{
	int object;
	int count;
};

class OcclusionGraph {
public:
	static vector<vector<OcclusionStruct>>Occludees;
	static vector<vector<OcclusionStruct>>OccludedBy;
	static void TraverseOcclusions();
	static void BuildGraph(DDS* dds);
	static set<int> GetOccluders(int object);
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

	vector<pointCoords> Coords;
	vector<pointColor> Colors;
	vector<int>ObjectIds;
	vector<float>Rads;	//constant for now//

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

		Rads.push_back(0.05);

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
	vector<bool>IsAnObjectID (maxobject[0] + 1, false);
	for (int i = 0; i < pnum; i++)
	{
		IsAnObjectID[ObjectIds[i]] = true;
	}
	objectsnum = 0;
	for (int i = 0; i < IsAnObjectID.size(); i++)
	{
		if (IsAnObjectID[i])
			objectsnum++;
	}
	MaxObject = maxobject[0];
	cout << "max: " << MaxObject << "\n";
	cout << "number of objects: " << objectsnum << "\n";
	////
	OrderedIDs.resize(MaxObject + 1);
	int index = 0;
	for (int i = 0; i < IsAnObjectID.size(); i++)
	{
		if (IsAnObjectID[i])
		{
			OrderedIDs[i] = index;
			index++;
		}
		else
			OrderedIDs[i] = -1;
	}

	RealIDs.resize(objectsnum);
	index = 0;
	for (int i = 0; i < IsAnObjectID.size(); i++)
	{
		if (IsAnObjectID[i])
		{
			RealIDs[index] = i;
			index++;
		}
	}


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
	//glm::mat4 ViewMat = glm::translate(glm::vec3(-midx,-midy,-(minz+2.5*(maxz-midz))));
	glm::mat4 ViewMat = glm::translate(glm::vec3(-midx,-midy,-(minz+3*(maxz-midz))));
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

	OcclusionGraph::BuildGraph(dds);

	dds->FreeMemory(true);
	
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
	
	OcclusionGraph::TraverseOcclusions();


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
		glUniformMatrix4fv(glGetUniformLocation(PntRdr.GetHandle(), "pvm_matrix"), 1, GL_FALSE, glm::value_ptr(pvmMat));
		glDrawArrays(GL_POINTS, 0, pnum);
		

		glfwSwapBuffers(myWindow);
		glfwPollEvents(); 
	}


	return 1;
}

void OcclusionGraph::TraverseOcclusions()
{
	for (int i = 0; i < objectsnum; i++)
	//for (int i = 0; i < MaxObject; i++)
	{
		for (OcclusionStruct s : Occludees[i])
			cout << RealIDs[i] << " occludes " << RealIDs[s.object] << "\n";

	}
}

void OcclusionGraph::BuildGraph(DDS* dds)
{

	dds->GetOcclusions(); //use DDS to get occlusion data

	unsigned long long* occpairHost = new unsigned long long[dds->OcclusionsNum];
	cudaMemcpy(occpairHost, dds->occpairCompact, dds->OcclusionsNum * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	int* occpairCountHost = new int[dds->OcclusionsNum];
	cudaMemcpy(occpairCountHost, dds->occpairCompactCount, dds->OcclusionsNum * sizeof(int), cudaMemcpyDeviceToHost);

	Occludees.resize(objectsnum);
	OccludedBy.resize(objectsnum);

	for (int i = 0; i < dds->OcclusionsNum; i++)
	{

		unsigned long long ull = occpairHost[i];
		unsigned long long ullcopy = ull;

		ullcopy = ullcopy >> 32;
		int occluder = ullcopy;

		ullcopy = ull & 0x00000000FFFFFFFF;
		int occludee = ullcopy;

		occluder = OrderedIDs[occluder];
		occludee = OrderedIDs[occludee];

		int occCount = occpairCountHost[i];

		if (occluder >= objectsnum || occluder < 0 || occludee >= objectsnum || occludee < 0)
			cout << "pbm in values: " << occluder << " " << occludee << "\n";
		//;
		else
		{
			OcclusionStruct s1 = { occludee, occCount };
			OcclusionStruct s2 = { occluder, occCount };

			Occludees[occluder].push_back(s1);
			OccludedBy[occludee].push_back(s2);

			//cout << "occlusion values: " << occluder << " " << occludee << "\n";
		}
	}

}

set<int> OcclusionGraph::GetOccluders(int object)
{
	set<int> occluders;

	for (OcclusionStruct s : OccludedBy[object])
		occluders.insert(s.object);

	return occluders;
}


vector<vector<OcclusionStruct>> OcclusionGraph::Occludees;
vector<vector<OcclusionStruct>> OcclusionGraph::OccludedBy;
