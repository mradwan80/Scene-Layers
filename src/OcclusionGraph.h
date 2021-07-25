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
	vector<vector<OcclusionStruct>>Occludees;
	vector<vector<OcclusionStruct>>OccludedBy;

public:
	OcclusionGraph();
	void TraverseOcclusions();
	void BuildGraph(DDS* dds, int objectsnum);
	set<int> GetOccluders(int object);
	
};
