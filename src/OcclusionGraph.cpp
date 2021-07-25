#include "OcclusionGraph.h"

OcclusionGraph::OcclusionGraph(){}

void OcclusionGraph::TraverseOcclusions()
{
	for (int i = 0; i < OccludedBy.size(); i++)
	{
		for (OcclusionStruct s : Occludees[i])
			cout << i << " occludes " << s.object << "\n";

	}
}

void OcclusionGraph::BuildGraph(DDS* dds, int objectsnum)
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

		int occCount = occpairCountHost[i];

		if (occluder >= objectsnum || occluder < 0 || occludee >= objectsnum || occludee < 0)
			cout << "pbm in values: " << occluder << " " << occludee << "\n";
		else
		{
			OcclusionStruct s1 = { occludee, occCount };
			OcclusionStruct s2 = { occluder, occCount };

			Occludees[occluder].push_back(s1);
			OccludedBy[occludee].push_back(s2);

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