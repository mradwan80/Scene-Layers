#pragma once

//x:pixel
//p:pile
//f:frag
//t:patch

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <thrust/host_vector.h>

#include <vector>
using namespace std;

enum PileSampleType {AVG, START};

struct pointCoords
{
	float x, y, z;
};

struct pointNormal
{
	float x, y, z;
};

struct pointColor
{
	float r, g, b;
};

class PileStruct
{
public:
	int pixel;
	int genTriangle;
	float start, end;
	float depth;
	int pilePatch;
	bool istop;
};

class FragStruct
{
public:
	int vertexInd;
	float depth;
};

class DDS
{

private:

	PileSampleType SampleValue;
	bool Debug_Mode;

	cudaEvent_t startE, stopE;

	vector<pointCoords>*vxPos;
	vector<float>* vxRad;
	vector<int>*vxPatch;

	float viewWidth;
	glm::mat4 vmMat, pvmMat, pvmOrthoMat;
	glm::mat4 projectionMat;
	
	int FragsNum;
	int PixelPatchNum;

	float* matrixPVM;
	float* matrixVM;
	float* vpos;
	float* vrad;
	int* vptc;
	
	float* FragDepth; float* FragRad; int* FragVertex; int* FragDist; unsigned long long* FragPixelPatch;
	//unsigned long long* PixelPatch;
	float* pstartBig;	float* pdepthBig;  int* pvertexBig; int* ppixelBig; int* ptopBig; unsigned long long* pkeyBig; //size FragsNum
	float* pstart;	float* pdepth;  int* pvertex; int* ppixel; int* ptop; unsigned long long* pkey;

	int* xfcount;	int* xfoffset;
	int* xtfcount; int* xtfoffset;
	

	void PrepareInput();
	void CountFrags();
	void ProjectFrags();
	void SortFrags();
	void Pile();
	void FindTopPiles();
	void CountPiles();
	void CleanPiles();
	void SortPiles();
	
	void TestCountFrags();
	//void TestCreateOffset();
	void TestProjectFrags();
	void TestSortFrags();
	void TestPile();
	void TestCountAndCleanPiles();
	void TestSortPiles();

	//occlusions related buffer//
	int* xocount; int* xooffset;
	int* xpcount; int* xpoffset;
	int* xpcountBig;
	unsigned long long* occpair; //initial. repetitions.
	unsigned long long* occpairCompactBig; //compact, but big size, rest are zeros//
	int* occpairCompactCountBig;

	float PrepareTime = 0;
	float CountFragsTime = 0;
	float ProjectFragsTime = 0;
	float SortFragsTime = 0;
	float PileTime = 0;
	float FindTopPileTime = 0;
	float CleanPilesTime = 0;
	float CountPilesTime = 0;
	float SortPilesTime = 0;
	float FinalizePilesTime = 0;
	float FreeMemoryTime = 0;
	float OverallTime = 0;


public:

	int globalW, globalH;

	int PilesNum;
	vector<int> PilesCount;
	vector<int> PilesOffset;
	vector<PileStruct> Piles;
	vector<int>PilesVertex;
	vector<bool>PilesFacing;
	vector<int>VertexPile;	//useful for navigation
	vector<float>VertexZ;

	int OcclusionsNum;
	unsigned long long* occpairCompact;
	int* occpairCompactCount;

	DDS();
	DDS(PileSampleType SampleValue, bool Debug_Mode, int w, int h, float viewWidthI, vector<pointCoords>* Pos, vector<float>* Rad, vector<int>* Ptc, glm::mat4 projectionMatI, glm::mat4 vmMatI, glm::mat4 pvmMatI, glm::mat4 pvmOrthoMatI);
	
	void BuildDDS();
	void FreeMemory(bool FreeGraphBuffer);

	void FinalizePiles();

	void GetOcclusions();

	void OutputTimings();

	bool PilesIntersect(int p1, int p2);
	
};

void FillAllWithValue(int* arr, int sz, int val);
void FillAllWithValue(unsigned int* arr, int sz, unsigned int val);
void FillAllWithValue(float* arr, int sz, float val);
void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val);
void FillAllWithValue(bool* arr, int sz, bool val);
//
void CountFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* vptc, int* xfcount);
void SetPixelFragsOffsetCuda(int pxNum, int* xfcount, int* xfoffset);
int GetFragsNumCuda(int vxNum, int* xfcount);
//
void ProjectFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* vptc, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch);
//
void SortFragsCuda(int FragsNum, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch);
int GetPixelPatchNumCuda(int FragsNum, unsigned long long* FragPixelPatch);
void SetPixelPatchFragsCountAndOffset(int FragsNum, int PixelPatchNum, unsigned long long* FragPixelPatch, int* xtfcount, int* xtfoffset);
//
//void PileCuda(PileSampleType SampleValue, int PixelPatchNum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, unsigned long long* pkeyBig);
void PileCuda(PileSampleType SampleValue, int PixelPatchNum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, int* ptopBig, unsigned long long* pkeyBig);
void FindTopPilesCuda(int FragsNum, int* pvertexBig, int* ppixelBig, int* ptopBig, int* vptc);
int CountPilesCuda(int FragsNum, float* pstartBig);
void CleanPilesCuda(int FragsNum, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, int* ptopBig, unsigned long long* pkeyBig, float* pstart, float* pdepth, int* pvertex, int* ppixel, int* ptop, unsigned long long* pkey);
void SortPilesCuda(int FragsNum, float* pstart, float* pdepth, int* pvertex, int* ppixel, int* ptop, unsigned long long* pkey);


///occlusion relations functions///
void CountOcclusionsCuda(int PixelsNum, int* xpcount, int* xpoffset, float* pstart, float* pdepth, int* ptop, int* xocount);
void FindOcclusionsCuda(int PixelsNum, int* xpcount, int* xpoffset, float* pstart, float* pdepth, int* pvertex, int* ptop, int* vptc, int* xooffset, unsigned long long* occpair);
int SetPileCountBigCuda(int PilesNum, int PixelsNum, int* ppixel, int* xpcountBig);
void SetPileCountAndOffsetCuda(int PixelsNum, int FPixelsNum, int* xpcountBig, int* xpcount, int* xpoffset);
int GetOcclusionsNumCuda(int FPixelsNum, int* xocount);
void SetOcclusionOffsetCuda(int FPixelsNum, int* xocount, int* xooffset);
//to compress buffers//
void SetOccpairCompactAndCountBigCuda(int OcclusionsNum, unsigned long long* occpair, unsigned long long* occpairCompactBig, int* occpairCompactCountBig);
int GetCompactOcclusionsNumCuda(int OcclusionsNum, int* occpairCompactCountBig);
void SetOccpairCompactAndCountCuda(int OcclusionsNum, unsigned long long* occpairCompactBig, int* occpairCompactCountBig, unsigned long long* occpairCompact, int* occpairCompactCount);
