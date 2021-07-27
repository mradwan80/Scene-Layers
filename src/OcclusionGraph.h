#pragma once

#include "DDS.h"
#include <vector>
#include <set>

struct OcclusionStruct
{
	int object;
	int count;
};

class OcclusionGraph {

private:
    static OcclusionGraph *instance;

	vector<vector<OcclusionStruct>>Occludees;
	vector<vector<OcclusionStruct>>OccludedBy;

public:
    static OcclusionGraph *getInstance();
    
	//OcclusionGraph();
	void TraverseOcclusions();
	void BuildGraph(DDS* dds, int objectsnum);
	set<int> GetOccluders(int object);

	void DestroyInstance();
	
};
