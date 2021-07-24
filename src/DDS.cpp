#include "DDS.h"
#include <chrono>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include <thrust/execution_policy.h>
//#include "ClassesDefinitions.h"
using namespace std;

DDS::DDS() {}

DDS::DDS(PileSampleType SampleValue, bool Debug_Mode, int w, int h, float viewWidthI, vector<pointCoords>*Pos, vector<float>* Rad, vector<int>*Ptc, glm::mat4 projectionMatI, glm::mat4 vmMatI, glm::mat4 pvmMatI, glm::mat4 pvmOrthoMatI)
{
	this->SampleValue = SampleValue;
	this->Debug_Mode = Debug_Mode;

	globalW = w;
	globalH = h;
	viewWidth = viewWidthI;

	vxPos = Pos;
	vxRad = Rad;
	vxPatch = Ptc;

	vmMat = vmMatI;
	pvmMat = pvmMatI;
	pvmOrthoMat = pvmOrthoMatI;
	projectionMat = projectionMatI;

}

void DDS::PrepareInput()
{
	float milliseconds;

	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	//allocate
	cudaMalloc((void**)&matrixVM, 16 * sizeof(float));
	cudaMalloc((void**)&matrixPVM, 16 * sizeof(float));
	cudaMalloc((void**)&vpos, vxPos->size() * 3 * sizeof(float));
	cudaMalloc((void**)&vrad, vxPos->size() * sizeof(float));
	cudaMalloc((void**)&vptc, vxPos->size() * sizeof(int));
	cudaMalloc((void**)&xfcount, globalW * globalH * sizeof(int));
	
	//copy
	cudaMemcpy(matrixVM, (float*)glm::value_ptr(vmMat), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(matrixPVM, (float*)glm::value_ptr(pvmMat), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vpos, vxPos->data(), vxPos->size() * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vrad, vxRad->data(), vxRad->size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vptc, vxPatch->data(), vxPatch->size() * sizeof(int), cudaMemcpyHostToDevice);
	
	FillAllWithValue(xfcount, globalW * globalH, 0);

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);
	
	PrepareTime = milliseconds;

}

void DDS::CountFrags()
{
	float milliseconds;

	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);
	
	CountFragsCuda(vxPos->size(), globalW, globalH, viewWidth, matrixVM, matrixPVM, vpos, vrad, vptc, xfcount);

	//allocate
	cudaMalloc((void**)&xfoffset, globalW * globalH * sizeof(int));

	//set values
	SetPixelFragsOffsetCuda(globalW * globalH, xfcount, xfoffset);

	FragsNum = GetFragsNumCuda(globalW * globalH, xfcount);
	//if (Debug_Mode) cout << "fragsnum: " << FragsNum << '\n';

	cudaMalloc((void**)&FragDepth, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragRad, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragVertex, FragsNum * sizeof(int));
	cudaMalloc((void**)&FragDist, FragsNum * sizeof(int));
	cudaMalloc((void**)&FragPixelPatch, FragsNum * sizeof(unsigned long long));

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	CountFragsTime = milliseconds;

}

void DDS::TestCountFrags()
{
	//get vector back
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	
	//make sure all either 1s or 0s
	int ctr = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		/*if (xfcountHost[i] == 1)
			ctr++;
		else if (xfcountHost[i] == 0)
		{

		}
		else
			cout << "incorrect value " << xfcountHost[i] << " at " << i << '\n';*/
		if (xfcountHost[i] > 0)
			ctr += xfcountHost[i];

		//if (xfcountHost[i] != 0)
		//	cout << "value in xfcountHost is: " << xfcountHost[i] << '\n';
	}
	
	//make sure 1s equal to vxs num
	cout << "sizes: " << vxPos->size() << ' ' << ctr << '\n';

	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	//int ctr = xfcountHost[0];
	ctr = 0;
	for (int i = 1; i < globalW * globalH; i++)
	{
		ctr += xfcountHost[i - 1];
		if (xfoffsetHost[i] != ctr)
			cout << "problem at " << i << '\n';
	}

	cout << "offset test: " << vxPos->size() << ' ' << ctr << '\n'; //equal when one frag per vertex//

}

void DDS::ProjectFrags()
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	//zero count again
	//cudaDeviceSynchronize();
	FillAllWithValue(xfcount, globalW * globalH, 0);
	//cudaDeviceSynchronize();
	//cudaDeviceSynchronize();

	
	//project using vbo data, count and offset
	ProjectFragsCuda(vxPos->size(), globalW, globalH, viewWidth, matrixVM, matrixPVM, vpos, vrad, vptc, xfcount, xfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch);

	cudaFree(vpos);
	cudaFree(vrad);
	//cudaFree(vptc);
	cudaFree(xfcount);
	cudaFree(xfoffset);

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	ProjectFragsTime = milliseconds;
}

void DDS::TestProjectFrags()
{
	int newSum = GetFragsNumCuda(globalW * globalH, xfcount);
	cout << "fragsnum: " << newSum << ' ' << FragsNum << '\n';
}

void DDS::SortFrags()
{

	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	SortFragsCuda(FragsNum, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch);

	PixelPatchNum = GetPixelPatchNumCuda(FragsNum, FragPixelPatch);

	cudaMalloc((void**)&xtfcount, PixelPatchNum * sizeof(int));
	cudaMalloc((void**)&xtfoffset, PixelPatchNum * sizeof(int)); //allocate, with PixelPatchpNum

	SetPixelPatchFragsCountAndOffset(FragsNum, PixelPatchNum, FragPixelPatch, xtfcount, xtfoffset);

	//if (Debug_Mode) cout << "(patch,pixel) Num: " << PixelPatchNum << '\n';

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	SortFragsTime = milliseconds;
}

void DDS::TestSortFrags()
{

	/////////////
	//check (patch,pixel) are ordered. check depths are ordered//
	/////////////

	float* fragdepthHost = new float[FragsNum];
	cudaMemcpy(fragdepthHost, FragDepth, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	unsigned long long* fragpixelpatchHost = new unsigned long long[FragsNum];
	cudaMemcpy(fragpixelpatchHost, FragPixelPatch, FragsNum * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 50; i++)
	{
		cout << fragdepthHost[i] << " " << fragpixelpatchHost[i] << '\n';
	}

	
	bool pxlproblem = false;
	bool dstproblem = false;
	float prevf = fragdepthHost[0];
	unsigned long long prevull = fragpixelpatchHost[0];
	for (int i = 1; i < FragsNum; i++)
	{
		float f = fragdepthHost[i];
		unsigned long long ull = fragpixelpatchHost[i];
		if (ull != prevull)
		{
			if (ull < prevull)
			{
				cout << "problem in the patch,pixel order at " << i - 1 << " and " << i << '\n';
				pxlproblem = true;
			}
		}
		else
		{
			if (f < prevf)
			{
				cout << "problem in the depth order at " << i - 1 << " and " << i << '\n';
				cout << "values are " << prevull << ' ' << ull << ' ' << prevf << ' ' << f << '\n';
				dstproblem = true;
			}
		}

		prevull = ull;
		prevf = f;
	}
	if (!pxlproblem)
		cout << "no problems found in the sorted patch,pixel list\n";
	if (!dstproblem)
		cout << "no problems found in the sorted depth list\n";
	
	/////////////
	//check all pixels with frags, exist in FragPixelPatch after sort//
	/////////////

	//get old count//
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	

	vector<bool> pixelFilled(globalW * globalH,false);
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (xfcountHost[i] > 0)
			pixelFilled[i] = true;
	}

	vector<bool> pixelFilled2(globalW * globalH, false);
	for (int i = 0; i < FragsNum; i++)
	{
		unsigned long long pixelpatch = fragpixelpatchHost[i];
		pixelpatch = pixelpatch >> 32;
		int pxl = pixelpatch; //get it
		pixelFilled2[pxl] = true;
	}

	for (int i = 0; i < globalW * globalH; i++)
	{
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
			cout << "problem at pixel " << i << '\n';
	}

	int ccc1 = 0, ccc2 = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (pixelFilled[i])
			ccc1++;
		if (pixelFilled2[i])
			ccc2++;
	}
	cout << "pixels filled: " << ccc1 << ' ' << ccc2 << '\n';

	////////
	//check that PixelPatchs are correctly counted in xtfcount//
	////////


	//get xtfcount
	int* xtfcountHost = new int[PixelPatchNum];
	cudaMemcpy(xtfcountHost, xtfcount, PixelPatchNum * sizeof(int), cudaMemcpyDeviceToHost);
	
	unsigned long long prevpixelpatch = fragpixelpatchHost[0];
	int ctr = 1;
	int index = 0;
	bool pbmFound = false;
	for (int i = 1; i < FragsNum; i++)
	{
		unsigned long long pixelpatch = fragpixelpatchHost[i];

		if (pixelpatch == prevpixelpatch)
		{
			ctr++;
		}
		else
		{
			if (xtfcountHost[index] != ctr)
			{
				cout << "problem: " << xtfcountHost[index] << ' ' << ctr << "\n";
				pbmFound = true;
			}

			prevpixelpatch = pixelpatch;
			ctr = 1;
			index++;
		}
		
	}
	cout << "ptc pxl nums: " << index << ' ' << PixelPatchNum << '\n';
	if (!pbmFound)
		cout << "counts of pixelpatchs are correct\n";

	////////
	//check xtfcount and xtfoffset are coherent//
	////////

	int* xtfoffsetHost = new int[PixelPatchNum];
	cudaMemcpy(xtfoffsetHost, xtfoffset, PixelPatchNum * sizeof(int), cudaMemcpyDeviceToHost);

	//int ctrr = xtfcountHost[0];
	int ctrr = 0;
	pbmFound = false;
	for (int i = 1; i < PixelPatchNum; i++)
	{
		ctrr += xtfcountHost[i - 1];
		if (xtfoffsetHost[i] != ctrr)
		{
			cout << "problem at " << i << " with " << ctrr << " and " << xtfoffsetHost[i] << '\n';
			pbmFound = true;
		}
	}

	cout << "xtfoffset test: " << ctrr + xtfcountHost[PixelPatchNum - 1] << '\n'; //equal when one frag per vertex//
	if (!pbmFound)
		cout << "pp counts and offsets are correct\n";


	////////
	//check pcount and xtfoffset reference FragPatch{Pixel correctly (by checking all pixels with counts exist in FragPixelPatch)//
	////////

	vector<bool> pixelFilled3(globalW * globalH, false);
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (xfcountHost[i] > 0)
			pixelFilled3[i] = true;
	}

	vector<bool> pixelFilled4(globalW * globalH, false);
	for (int i = 0; i < PixelPatchNum; i++)
	{
		int count = xtfcountHost[i];
		int offset = xtfoffsetHost[i];

		for (int j = offset; j < offset + count; j++)
		{
			unsigned long long ull = fragpixelpatchHost[j] >> 32;
			int pxl = ull;

			if (pxl<0 || pxl>globalW * globalH - 1)
				cout << "(TestSortFrags: we have a problem. a pixel of " << pxl << '\n';
			//if(pxl!=0)
			if (pxl > 0 && pxl < globalW * globalH)
				pixelFilled4[pxl] = true;
		}
	}

	int ccc3 = 0, ccc4 = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (pixelFilled3[i])
			ccc3++;
		if (pixelFilled4[i])
			ccc4++;
	}
	cout << "pixels filled: " << ccc3 << ' ' << ccc4 << '\n';
}


void DDS::Pile()
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	//allocate pstartBig, pvertexBig, ppixelBig (frags size)//
	cudaMalloc((void**)&pstartBig, FragsNum * sizeof(float));
	cudaMalloc((void**)&pdepthBig, FragsNum * sizeof(float));
	cudaMalloc((void**)&pvertexBig, FragsNum * sizeof(int));
	cudaMalloc((void**)&ppixelBig, FragsNum * sizeof(int));
	cudaMalloc((void**)&ptopBig, FragsNum * sizeof(int));
	cudaMalloc((void**)&pkeyBig, FragsNum * sizeof(unsigned long long));

	FillAllWithValue(pstartBig, FragsNum, -1.0);
	FillAllWithValue(pdepthBig, FragsNum, -1.0);
	FillAllWithValue(pvertexBig, FragsNum, -1);
	FillAllWithValue(ppixelBig, FragsNum, -1);
	FillAllWithValue(ptopBig, FragsNum, -1);
	FillAllWithValue(pkeyBig, FragsNum, 0xFFFFFFFFFFFFFFFF);
	
	//PileCuda(SampleValue, PixelPatchNum, xtfcount, xtfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch, pstartBig, pdepthBig, pvertexBig, ppixelBig, pkeyBig);
	PileCuda(SampleValue, PixelPatchNum, xtfcount, xtfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch, pstartBig, pdepthBig, pvertexBig, ppixelBig, ptopBig, pkeyBig);

	cudaFree(FragDepth);
	cudaFree(FragRad);
	cudaFree(FragVertex);
	cudaFree(FragDist);
	cudaFree(FragPixelPatch);

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	PileTime = milliseconds;
}

void DDS::TestPile()
{
	////////
	//check output in ppixel is correct (all pixels with frags, exist in ppixel)//
	////////

	/*
	//get old count//
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	int* ppixelHost = new int[FragsNum];
	cudaMemcpy(ppixelHost, ppixelBig, FragsNum * sizeof(int), cudaMemcpyDeviceToHost);

	vector<bool> pixelFilled(globalW * globalH, false);
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (xfcountHost[i] > 0)
			pixelFilled[i] = true;
	}

	vector<bool> pixelFilled2(globalW * globalH, false);
	for (int i = 0; i < FragsNum; i++)
	{
		int pxl = ppixelHost[i];
		//if (pxl<0 || pxl>globalW * globalH - 1)
		//	cout << "Test Pile: we have a problem. a pixel of " << pxl <<'\n'; //incorrect test. many memory locations will contain -1s
		
		if (pxl >= 0 && pxl < globalW * globalH)
			pixelFilled2[pxl] = true;
	}

	bool pbmFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
		{
			cout << "problem at pixel " << i << '\n';
			pbmFound = true;
		}
	}
	if (!pbmFound)
		cout << "all pixels with frags have piles\n";

	int ccc1 = 0, ccc2 = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (pixelFilled[i])
			ccc1++;
		if (pixelFilled2[i])
			ccc2++;
	}
	cout << "pixels filled (with frags, with piles): " << ccc1 << ' ' << ccc2 << '\n';

	////////
	//check start is always less than end//
	////////

	float* pstartHost = new float[FragsNum];
	cudaMemcpy(pstartHost, pstartBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pendHost = new float[FragsNum];
	cudaMemcpy(pendHost, pendBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);

	pbmFound = false;
	for (int i = 0; i < FragsNum; i++)
	{
		float start = pstartHost[i];
		float end = pendHost[i];

		if (start == -1.0 && end == -1.0)
			continue;
		else if (start > end)
		{
			cout << "start is bigger than end at index " << i << '\n';
			pbmFound = true;
		}
	}
	if (!pbmFound)
		cout << "starts are less than ends in all piles\n";
	
	////////
	//check avg depth is always between start and end//
	////////

	float* pdepthHost = new float[FragsNum];
	cudaMemcpy(pdepthHost, pdepthBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);

	pbmFound = false;
	for (int i = 0; i < FragsNum; i++)
	{
		float start = pstartHost[i];
		float end = pendHost[i];
		float depth = pdepthHost[i];

		if (start == -1.0 && end == -1.0 && depth == -1.0)
			continue;
		else if (depth < start || depth > end)
		{
			cout << "avg depth is not between start and end at index " << i << '\n';
			pbmFound = true;
		}
	}
	if (!pbmFound)
		cout << "avg depths are between starts and ends in all piles\n";
		*/
}

void DDS::FindTopPiles()
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);


	FindTopPilesCuda(FragsNum, pvertexBig, ppixelBig, ptopBig, vptc);

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	FindTopPileTime = milliseconds;
}

void DDS::CountPiles()
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);


	PilesNum = CountPilesCuda(FragsNum, pstartBig);

	//if (Debug_Mode) cout << "Piles number: " << PilesNum << '\n';


	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	CountPilesTime = milliseconds;
}

void DDS::CleanPiles()
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	cudaMalloc((void**)&pstart, PilesNum * sizeof(float));
	cudaMalloc((void**)&pdepth, PilesNum * sizeof(float));
	cudaMalloc((void**)&pvertex, PilesNum * sizeof(int));
	cudaMalloc((void**)&ppixel, PilesNum * sizeof(int));
	cudaMalloc((void**)&ptop, PilesNum * sizeof(int));
	cudaMalloc((void**)&pkey, PilesNum * sizeof(unsigned long long));


	//CleanPilesCuda(FragsNum, pstartBig, pdepthBig, pvertexBig, ppixelBig, pkeyBig, pstart, pdepth, pvertex, ppixel, pkey);
	CleanPilesCuda(FragsNum, pstartBig, pdepthBig, pvertexBig, ppixelBig, ptopBig, pkeyBig, pstart, pdepth, pvertex, ppixel, ptop, pkey);

	cudaFree(pstartBig);
	cudaFree(pdepthBig);
	cudaFree(pvertexBig);
	cudaFree(ppixelBig);
	cudaFree(ptopBig);
	cudaFree(pkeyBig);

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	CleanPilesTime = milliseconds;
}

void DDS::TestCountAndCleanPiles()
{
	// **important** make sure you don't cudaFree in previous function //

	//get all old 5
	//get all new 5
	//loop and check

	/*
	float* pstartHostB = new float[FragsNum];
	cudaMemcpy(pstartHostB, pstartBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pendHostB = new float[FragsNum];
	cudaMemcpy(pendHostB, pendBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pdepthHostB = new float[FragsNum];
	cudaMemcpy(pdepthHostB, pdepthBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	int* pvertexHostB = new int[FragsNum];
	cudaMemcpy(pvertexHostB, pvertexBig, FragsNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* ppixelHostB = new int[FragsNum];
	cudaMemcpy(ppixelHostB, ppixelBig, FragsNum * sizeof(int), cudaMemcpyDeviceToHost);
	unsigned long long* pkeyHostB = new unsigned long long[FragsNum];
	cudaMemcpy(pkeyHostB, pkeyBig, FragsNum * sizeof(unsigned long long), cudaMemcpyDeviceToHost);


	float* pstartHost = new float[PilesNum];
	cudaMemcpy(pstartHost, pstart, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pendHost = new float[PilesNum];
	cudaMemcpy(pendHost, pend, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pdepthHost = new float[PilesNum];
	cudaMemcpy(pdepthHost, pdepth, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	int* pvertexHost = new int[PilesNum];
	cudaMemcpy(pvertexHost, pvertex, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* ppixelHost = new int[PilesNum];
	cudaMemcpy(ppixelHost, ppixel, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	unsigned long long* pkeyHost = new unsigned long long[PilesNum];
	cudaMemcpy(pkeyHost, pkey, PilesNum * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	bool pbmFound = false;
	int index = 0;
	for (int i = 0; i < FragsNum; i++)
	{
		if (pstartHostB[i] == -1.0)
		{
			if (pendHostB[i] != -1.0)
			{
				cout << "pendBig should be -1.0 at frag " << i << '\n';
				pbmFound = true;
			}

			if (pdepthHostB[i] != -1.0)
			{
				cout << "pdepthBig should be -1.0 at frag " << i << '\n';
				pbmFound = true;
			}

			if (pvertexHostB[i] != -1)
			{
				cout << "pvertexBig should be -1 at frag " << i << '\n';
				pbmFound = true;
			}
			if (ppixelHostB[i] != -1)
			{
				cout << "ppixelBig should be -1 at frag " << i << '\n';
				pbmFound = true;
			}
			if (pkeyHostB[i] != 0xFFFFFFFFFFFFFFFF)
			{
				cout << "pkeyBig should be MAX at frag " << i << '\n';
				pbmFound = true;
			}
		}
		else
		{
			if (pstartHostB[i] != pstartHost[index])
			{
				cout << "start equality problem at frag " << i << '\n';
				pbmFound = true;
			}
			if (pendHostB[i] != pendHost[index])
			{
				cout << "end equality problem at frag " << i << '\n';
				pbmFound = true;
			}
			if (pdepthHostB[i] != pdepthHost[index])
			{
				cout << "depth equality problem at frag " << i << '\n';
				pbmFound = true;
			}
			if (pvertexHostB[i] != pvertexHost[index])
			{
				cout << "vertex equality problem at frag " << i << '\n';
				pbmFound = true;
			}
			if (ppixelHostB[i] != ppixelHost[index])
			{
				cout << "pixel equality problem at frag " << i << '\n';
				pbmFound = true;
			}
			if (pkeyHostB[i] != pkeyHost[index])
			{
				cout << "key equality problem at frag " << i << '\n';
				pbmFound = true;
			}
			index++;
		}
	}

	if (!pbmFound)
		cout << "shrinking piles arrays done successfully\n";
	cout << "piles num (in test fnc): " << PilesNum << ' ' << index << '\n';

	*/
}

void DDS::SortPiles()
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	//SortPilesCuda(PilesNum, pstart, pdepth, pvertex, ppixel, pkey);
	SortPilesCuda(PilesNum, pstart, pdepth, pvertex, ppixel, ptop, pkey);

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	SortPilesTime = milliseconds;
}

void DDS::TestSortPiles()
{
	/*
	float* pstartHost = new float[PilesNum];
	cudaMemcpy(pstartHost, pstart, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pendHost = new float[PilesNum];
	cudaMemcpy(pendHost, pend, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	int* pvertexHost = new int[PilesNum];
	cudaMemcpy(pvertexHost, pvertex, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* ppixelHost = new int[PilesNum];
	cudaMemcpy(ppixelHost, ppixel, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	

	//for (int i = 0; i <100 ; i++)
	//{
	//	float start = pstartHost[i];
	//	float end = pendHost[i];
	//	int pixel = ppixelHost[i];

	//	cout << pixel << ' ' << start << ' ' << end << '\n';
	//}

	bool pbmFound;
	int prepixel;

	
	//pbmFound = false;
	//float prestart = pstartHost[0];
	//float preend = pendHost[0];
	//int prevertex = pvertexHost[0];
	//prepixel = ppixelHost[0];
	//for (int i = 1; i < PilesNum; i++)
	//{
	//	float start = pstartHost[i];
	//	float end = pendHost[i];
	//	int vertex = pvertexHost[i];
	//	int pixel = ppixelHost[i];

	//	if (pixel == prepixel)
	//	{
	//		
	//		if (start < preend && Model::vxPatch[vertex] == Model::vxPatch[prevertex])
	//		{
	//			cout << "start of pile intersects previous pile\n";
	//			cout << prepixel << ' ' << prestart << ' ' << preend << ' ' << pixel << ' ' << start << ' ' << end << '\n';
	//			pbmFound = true;
	//		}
	//		if (start < prestart)
	//		{
	//			cout << "start of pile precedes start of previous pile\n";
	//			cout << prepixel << ' ' << prestart << ' ' << preend << ' ' << pixel << ' ' << start << ' ' << end << '\n';
	//			pbmFound = true;
	//		}
	//			
	//	}

	//	prestart = start;
	//	preend = end;
	//	prevertex = vertex;
	//	prepixel = pixel;
	//	
	//}
	//if (!pbmFound)
	//	cout << "starts and ends of piles are ok\n";

	pbmFound = false;
	prepixel = ppixelHost[0];
	for (int i = 1; i < PilesNum; i++)
	{
		int pixel = ppixelHost[i];

		if (pixel < prepixel)
			cout << "pixel order problem at frag " << i << ", pixels: " << prepixel << ' ' << pixel << '\n';

		if (pixel < prepixel)
			pbmFound = true;

		prepixel = pixel;

	}

	if (!pbmFound)
		cout << "pixels are correctly ordered\n";
		*/
}

void DDS::FinalizePiles()
{
	float milliseconds;
	//cudaEvent_t starte, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);


	PilesCount.resize(globalW * globalH, 0);
	PilesOffset.resize(globalW * globalH);
	Piles.resize(PilesNum);
	PilesVertex.resize(PilesNum);

	float* pstartHost = new float[PilesNum];
	cudaMemcpy(pstartHost, pstart, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pdepthHost = new float[PilesNum];
	cudaMemcpy(pdepthHost, pdepth, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
	int* pvertexHost = new int[PilesNum];
	cudaMemcpy(pvertexHost, pvertex, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* ppixelHost = new int[PilesNum];
	cudaMemcpy(ppixelHost, ppixel, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* ptopHost = new int[PilesNum];
	cudaMemcpy(ptopHost, ptop, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);

	int prepixel = -1;
	PilesOffset[0] = 0;
	float start, depth; int vertex, pixel;
	int offset;
	int istop;
	for (int i = 0; i < PilesNum; i++)
	{
		start = pstartHost[i];
		depth = pdepthHost[i];
		vertex = pvertexHost[i];
		pixel = ppixelHost[i];
		istop = ptopHost[i];

		PilesCount[pixel]++;
		Piles[i].start = start;
		Piles[i].end = start + 2 * (depth - start);
		Piles[i].depth = depth;
		Piles[i].pixel = pixel;
		Piles[i].istop = (istop == 1);
		PilesVertex[i] = vertex;

		if (pixel != prepixel && prepixel != -1)
		{
			offset = PilesOffset[prepixel] + PilesCount[prepixel];
			PilesOffset[pixel] = offset;

			//if want to fill PilesOffset of all pixels between prepixel and pixel (if any), can use this. but probably unimportant (their PilesCount is 0 anyway)
			//for (int pxl = prepixel + 1; pxl <= pixel; pxl++)				
			//	PilesOffset[pxl] = offset;
		}

		prepixel = pixel;
	}
	
	//if want to fill PilesOffset of all pixels in [pixel+1,globalW*globalH), can use this. but probably unimportant (their PilesCount is 0 anyway)
	//for (int pxl = pixel + 1; pxl < globalW * globalH; pxl++)
	//	PilesOffset[pxl] = offset;

	delete[] pstartHost;
	delete[] pdepthHost;
	delete[] pvertexHost;
	delete[] ppixelHost;
	delete[] ptopHost;


	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	FinalizePilesTime = milliseconds;
}

void DDS::FreeMemory(bool FreeGraphBuffer)
{
	float milliseconds;
	//cudaEvent_t start, stop;
	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	cudaFree(matrixVM);
	cudaFree(matrixPVM);

	cudaFree(xtfcount);
	cudaFree(xtfoffset);

	cudaFree(pstart);
	cudaFree(pdepth);
	cudaFree(pvertex);
	cudaFree(ppixel);
	cudaFree(ptop);
	cudaFree(pkey);


	cudaFree(vptc);

	if (FreeGraphBuffer)
	{
		cudaFree(xpcountBig);
		cudaFree(xpcount);
		cudaFree(xpoffset);
		cudaFree(xocount);
		cudaFree(xooffset);
		cudaFree(occpair);
		cudaFree(occpairCompactBig);
		cudaFree(occpairCompactCountBig);
		cudaFree(occpairCompact);
		cudaFree(occpairCompactCount);
	}

	//cudaDeviceSynchronize();


	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);

	FreeMemoryTime = milliseconds;
}

void DDS::OutputTimings()
{
	
	cout << "***Preparation time: " << PrepareTime << '\n';
	cout << "***CountFrags time: " << CountFragsTime << '\n';
	cout << "***ProjectFrags time: " << ProjectFragsTime << '\n';
	cout << "***SortFrags time: " << SortFragsTime << '\n';
	cout << "***Pile time: " << PileTime<< '\n';
	cout << "***Find top pile time: " << FindTopPileTime << '\n';
	cout << "***CleanPiles time: " << CleanPilesTime << '\n';
	cout << "***CountPiles time: " << CountPilesTime << '\n';
	cout << "***SortPiles time: " << SortPilesTime << '\n';
	cout << "***FinalizePiles: " << FinalizePilesTime << '\n';
	cout << "***FreeMemory time: " << FreeMemoryTime << '\n';
	cout << "***Overall time: " << OverallTime << '\n';
}

void DDS::BuildDDS()
{
	
	float milliseconds;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	PrepareInput();
	CountFrags(); 
	//TestCountFrags();
	ProjectFrags();
	//TestProjectFrags();
	SortFrags(); 
	//TestSortFrags();
	Pile(); 
	//TestPile();
	FindTopPiles();
	CountPiles(); 
	CleanPiles(); 
	//TestCountAndCleanPiles();
	SortPiles(); 
	//TestSortPiles();
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	//this is not part of the DDS creation
	//FinalizePiles(); cout << "finalize done\n";

	//if (Debug_Mode) cout << "***total DDS time: " << milliseconds << '\n';

	//if (Debug_Mode) cout << "---------\n";

	OverallTime = milliseconds;
	
	
}

void DDS::GetOcclusions()
{
	float milliseconds;

	cudaEventCreate(&startE);
	cudaEventCreate(&stopE);

	cudaEventRecord(startE);

	//init pcountBig//
	cudaMalloc((void**)&xpcountBig, globalW * globalH * sizeof(int));
	FillAllWithValue(xpcountBig, globalW * globalH, -1);


	//get number of filled pixels (those with piles)//
	int FPixelsNum = SetPileCountBigCuda(PilesNum, globalW * globalH, ppixel, xpcountBig);

	//init pcount and poffset//
	cudaMalloc((void**)&xpcount, FPixelsNum * sizeof(int));
	cudaMalloc((void**)&xpoffset, FPixelsNum * sizeof(int));

	//fill pcount and poffset//
	SetPileCountAndOffsetCuda(globalW * globalH, FPixelsNum, xpcountBig, xpcount, xpoffset);

	//if (Debug_Mode) cout << "filled pixels num: " << FPixelsNum << '\n';

	//allocate and set xocount (occlusions per pixel)//
	cudaMalloc((void**)&xocount, FPixelsNum * sizeof(int));
	FillAllWithValue(xocount, FPixelsNum, 0);
	CountOcclusionsCuda(FPixelsNum, xpcount, xpoffset, pstart, pdepth, ptop, xocount);

	//allocate and set ooffset//
	cudaMalloc((void**)&xooffset, FPixelsNum * sizeof(int));
	SetOcclusionOffsetCuda(FPixelsNum, xocount, xooffset);

	//calculate total occ pairs//
	OcclusionsNum = GetOcclusionsNumCuda(FPixelsNum, xocount);

	//if (Debug_Mode) cout << "Occlusions num: " << OcclusionsNum << "\n";


	//allocate and find occpairs//
	cudaMalloc((void**)&occpair, OcclusionsNum * sizeof(unsigned long long));
	FindOcclusionsCuda(FPixelsNum, xpcount, xpoffset, pstart, pdepth, pvertex, ptop, vptc, xooffset, occpair);



	//allocate and calculate Bigs//
	cudaMalloc((void**)&occpairCompactBig, OcclusionsNum * sizeof(unsigned long long));
	FillAllWithValue(occpairCompactBig, OcclusionsNum, 0);
	cudaMalloc((void**)&occpairCompactCountBig, OcclusionsNum * sizeof(int));
	FillAllWithValue(occpairCompactCountBig, OcclusionsNum, 0);
	SetOccpairCompactAndCountBigCuda(OcclusionsNum, occpair, occpairCompactBig, occpairCompactCountBig);


	//get new occlusions num//
	int NewOcclusionsNum = GetCompactOcclusionsNumCuda(OcclusionsNum, occpairCompactCountBig);

	//if (Debug_Mode) cout << "reduced occlusions num: " << NewOcclusionsNum << "\n";

	//allocate and calculate final//
	cudaMalloc((void**)&occpairCompact, NewOcclusionsNum * sizeof(unsigned long long));
	cudaMalloc((void**)&occpairCompactCount, NewOcclusionsNum * sizeof(int));
	SetOccpairCompactAndCountCuda(OcclusionsNum, occpairCompactBig, occpairCompactCountBig, occpairCompact, occpairCompactCount);

	OcclusionsNum = NewOcclusionsNum;

	cudaEventRecord(stopE);

	cudaEventSynchronize(stopE);
	cudaEventElapsedTime(&milliseconds, startE, stopE);
	//cout << "***Finding Occlusions time: " << milliseconds << '\n';


}

bool DDS::PilesIntersect(int p1, int p2)
{
	if (Piles.size() <= p1 || Piles.size() <= p2)
		return false;
	
	if ((Piles[p1].start <= Piles[p2].start && Piles[p2].start <= Piles[p1].end) || (Piles[p1].start <= Piles[p2].end && Piles[p2].end <= Piles[p1].end) || (Piles[p2].start <= Piles[p1].start && Piles[p1].end <= Piles[p2].end))
		return true;
	else 
		return false;
}