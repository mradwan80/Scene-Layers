#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include<thrust/sequence.h>
#include<thrust/gather.h>
#include<thrust/count.h>
#include <thrust/execution_policy.h>
#include<thrust/copy.h>
#include "DDS.h"

__global__ void CountFragsKernel(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* vptc, int* xfcount)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{
		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		float posZvm = vmMat[2] * x + vmMat[6] * y + vmMat[10] * z + vmMat[14] * w;

		if (posZvm >= 0) return;

		float fov = 60 * 3.14 / 180;
		float slope = tan(fov / 2.0);
		float projFactor = -0.5 * globalH / (slope * posZvm);

		int maxSplatSize = 25;
		//int maxSplatSize = 30;

		float worldSpaceSize = vrad[v] * 2 /5;
		int SplatSize = int(worldSpaceSize * projFactor);
		if(SplatSize> maxSplatSize)
			SplatSize = maxSplatSize;

		//int SplatSize = int(round(2.0 * (vrad[v] / viewWidth) * globalW));
		int SplatRad = SplatSize / 2;
		float SplatRadSqr = (SplatRad + 0.5) * (SplatRad + 0.5);


		int xstart = xscreen - SplatRad;
		int xend = xscreen + SplatRad;
		int ystart = yscreen - SplatRad;
		int yend = yscreen + SplatRad;
		/*if (SplatRad % 2 == 0)
		{
			xend--;
			yend--;
		}*/
		
		for (int x = xstart; x <= xend; x++)
		{
			if (x<0 || x>globalW - 1)
				continue;
			for (int y = ystart; y <= yend; y++)
			{
				if (y<0 || y>globalH - 1)
					continue;

				float xdiff = abs(x - xscreen);
				float ydiff = abs(y - yscreen);
				if (xdiff > 0) xdiff -= 0.5; if (ydiff > 0) ydiff -= 0.5;
				if (SplatRad > 1 && xdiff * xdiff + ydiff * ydiff > SplatRadSqr)
					continue;

				int pxl = x + y * globalW; //from x and y

				if (pxl >= 0 && pxl < globalW * globalH)
					atomicAdd(&xfcount[pxl], 1);
			}
		}


	}
}


void CountFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* vptc, int* xfcount)
{
	CountFragsKernel<<<vxNum/256 + 1, 256 >>>(vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, vrad, vptc, xfcount);
	//cudaDeviceSynchronize();
}


void SetPixelFragsOffsetCuda(int pxNum, int* xfcount, int* xfoffset)
{
	thrust::device_ptr<int> o = thrust::device_pointer_cast(xfoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xfcount);
	
	//call thrust function
	thrust::exclusive_scan(c, c + pxNum, o);
	//cudaDeviceSynchronize();
}

int GetFragsNumCuda(int vxNum, int* xfcount)
{
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xfcount);

	//get count of xfcount//
	int FragsNum = thrust::reduce(c, c + vxNum, (int)0, thrust::plus<int>());
	//cudaDeviceSynchronize();

	return FragsNum;
}

__device__
unsigned long long GeneratePixelPatchKey(int patch, int pixel)
{
	/*unsigned long long result = pixel;
	result = result << 32;

	//unsigned long long result=0;

	const int lineParameter = patch;
	//uint converted_key = *((uint *)&lineParameter);
	unsigned int converted_key = *((unsigned int*)&lineParameter);
	const unsigned int mask = ((converted_key & 0x80000000) ? 0xffffffff : 0x80000000);
	converted_key ^= mask;

	result |= (unsigned long long)(converted_key);

	return result;*/

	unsigned long long result = pixel;
	result = result << 32;

	
	const int lineParameter = patch;
	//uint converted_key = *((uint *)&lineParameter);
	unsigned int converted_key = *((unsigned int*)&lineParameter);
	
	result |= (unsigned long long)(converted_key);

	return result;

}

__global__ void ProjectFragsKernel(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* vptc, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{

		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);


		float posZvm = vmMat[2] * x + vmMat[6] * y + vmMat[10] * z + vmMat[14] * w;

		if (posZvm >= 0) return;

		float fov = 60 * 3.14 / 180;
		float slope = tan(fov / 2.0);
		float projFactor = -0.5 * globalH / (slope * posZvm);

		int maxSplatSize = 25;
		//int maxSplatSize = 30;

		float worldSpaceSize = vrad[v] * 2 / 5;
		int SplatSize = int(worldSpaceSize * projFactor);
		if (SplatSize > maxSplatSize)
			SplatSize = maxSplatSize;

		//int SplatSize = int(round(2.0 * (vrad[v] / viewWidth) * globalW));
		int SplatRad = SplatSize / 2;
		float SplatRadSqr = (SplatRad + 0.5) * (SplatRad + 0.5);
		float rad = vrad[v];
		float depth = -posZvm - rad;
		int ptc = vptc[v];

		int xstart = xscreen - SplatRad;
		int xend = xscreen + SplatRad;
		int ystart = yscreen - SplatRad;
		int yend = yscreen + SplatRad;
		/*if (SplatRad % 2 == 0)
		{
			xend--;
			yend--;
		}*/

		for (int x = xstart; x <= xend; x++)
		{
			if (x<0 || x>globalW - 1)
				continue;
			for (int y = ystart; y <= yend; y++)
			{
				if (y<0 || y>globalH - 1)
					continue;

				float xdiff = abs(x - xscreen);
				float ydiff = abs(y - yscreen);
				if (xdiff > 0) xdiff -= 0.5; if (ydiff > 0) ydiff -= 0.5;
				if (SplatRad > 1 && xdiff * xdiff + ydiff * ydiff > SplatRadSqr)
					continue;

				int pxl = x + y * globalW; //from x and y

				int index, offset;
				if (pxl >= 0 && pxl < globalW * globalH)
				{
					offset = xfoffset[pxl];
					index = atomicAdd(&xfcount[pxl], 1) + offset;
					
					FragDepth[index] = depth;
					FragRad[index] = rad;
					FragVertex[index] = v;
					FragDist[index] = abs(x - xscreen) + abs(y - yscreen);
					FragPixelPatch[index] = GeneratePixelPatchKey(ptc, pxl);

				}
			}
		}


	}

}

void ProjectFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* vptc, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch)
{
	
	ProjectFragsKernel<<<vxNum / 256 + 1, 256 >>>(vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, vrad, vptc, xfcount, xfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch);
	//cudaDeviceSynchronize();
}

void SortFragsCuda(int FragsNum, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch)
{
	//device pointers//
	thrust::device_ptr<float> fd = thrust::device_pointer_cast(FragDepth);
	thrust::device_ptr<float> fr = thrust::device_pointer_cast(FragRad);
	thrust::device_ptr<int> fv = thrust::device_pointer_cast(FragVertex);
	thrust::device_ptr<int> fs = thrust::device_pointer_cast(FragDist);
	thrust::device_ptr<unsigned long long> fpp = thrust::device_pointer_cast(FragPixelPatch);

	//tmp buffers for thrust::gather//
	float* FragFloatTmp;
	int* FragIntTmp;
	unsigned long long* FragUllTmp;
	thrust::device_ptr<float> fft;
	thrust::device_ptr<int> fit;
	thrust::device_ptr<unsigned long long> fullt;


	//init an index buffer//
	unsigned int* FragIndex;
	cudaMalloc((void**)&FragIndex, FragsNum * sizeof(unsigned int));
	thrust::device_ptr<unsigned int> fi = thrust::device_pointer_cast(FragIndex);
	thrust::sequence(fi, fi + FragsNum, 0);
	//cudaDeviceSynchronize();


	//sort depth and index//
	thrust::sort_by_key(fd, fd + FragsNum, fi);
	//cudaDeviceSynchronize();


	/////////////////
	//change all other arrays based on the sorted index//
	/////////////////

	//rad//
	cudaMalloc((void**)&FragFloatTmp, FragsNum * sizeof(float));
	fft = thrust::device_pointer_cast(FragFloatTmp);
	thrust::gather(fi, fi + FragsNum, fr, fft);
	cudaMemcpy(FragRad, FragFloatTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(FragFloatTmp);

	//vertex and dis//
	cudaMalloc((void**)&FragIntTmp, FragsNum * sizeof(int));
	fit = thrust::device_pointer_cast(FragIntTmp);
	thrust::gather(fi, fi + FragsNum, fv, fit);
	cudaMemcpy(FragVertex, FragIntTmp, FragsNum * sizeof(int), cudaMemcpyDeviceToDevice);
	thrust::gather(fi, fi + FragsNum, fs, fit);
	cudaMemcpy(FragDist, FragIntTmp, FragsNum * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaFree(FragIntTmp);

	//pixel/patch
	cudaMalloc((void**)&FragUllTmp, FragsNum * sizeof(unsigned long long));
	fullt = thrust::device_pointer_cast(FragUllTmp);
	thrust::gather(fi, fi + FragsNum, fpp, fullt);
	cudaMemcpy(FragPixelPatch, FragUllTmp, FragsNum * sizeof(unsigned long long), cudaMemcpyDeviceToDevice);
	cudaFree(FragUllTmp);


	//re-init the index buffer//
	thrust::sequence(fi, fi + FragsNum, 0);
	//cudaDeviceSynchronize();

	//stable sort index and patch,pixel//
	thrust::stable_sort_by_key(fpp, fpp + FragsNum, fi);
	//cudaDeviceSynchronize();


	/////////////////
	//change all other arrays based on the sorted index//
	/////////////////

	//depth and rad//
	cudaMalloc((void**)&FragFloatTmp, FragsNum * sizeof(float));
	fft = thrust::device_pointer_cast(FragFloatTmp);
	thrust::gather(fi, fi + FragsNum, fd, fft);
	cudaMemcpy(FragDepth, FragFloatTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);
	thrust::gather(fi, fi + FragsNum, fr, fft);
	cudaMemcpy(FragRad, FragFloatTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(FragFloatTmp);

	//vertex and dis//
	cudaMalloc((void**)&FragIntTmp, FragsNum * sizeof(int));
	fit = thrust::device_pointer_cast(FragIntTmp);
	thrust::gather(fi, fi + FragsNum, fv, fit);
	cudaMemcpy(FragVertex, FragIntTmp, FragsNum * sizeof(int), cudaMemcpyDeviceToDevice);
	thrust::gather(fi, fi + FragsNum, fs, fit);
	cudaMemcpy(FragDist, FragIntTmp, FragsNum * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaFree(FragIntTmp);


	cudaFree(FragIndex);

}

int GetPixelPatchNumCuda(int FragsNum, unsigned long long* FragPixelPatch)
{
	unsigned long long* PixelPatchUnique;
	cudaMalloc((void**)&PixelPatchUnique, FragsNum * sizeof(unsigned long long));
	thrust::device_ptr<unsigned long long> ppu = thrust::device_pointer_cast(PixelPatchUnique);

	//copy to ppu//
	cudaMemcpy(PixelPatchUnique, FragPixelPatch, FragsNum * sizeof(unsigned long long), cudaMemcpyDeviceToDevice);

	thrust::device_ptr<unsigned long long> new_end = thrust::unique(ppu, ppu + FragsNum);
	int PixelPatchNum = new_end - ppu;

	cudaFree(PixelPatchUnique);

	return PixelPatchNum;
}

void SetPixelPatchFragsCountAndOffset(int FragsNum, int PixelPatchNum, unsigned long long* FragPixelPatch, int* xtfcount, int* xtfoffset)
{
	thrust::device_ptr<unsigned long long> fpp = thrust::device_pointer_cast(FragPixelPatch);

	//a buffer of ones//
	int* FragOnes;
	cudaMalloc((void**)&FragOnes, FragsNum * sizeof(int));
	thrust::device_ptr<int> fo = thrust::device_pointer_cast(FragOnes);
	thrust::fill(fo, fo + FragsNum, 1);


	thrust::device_ptr<int> ppc = thrust::device_pointer_cast(xtfcount);

	//dummy buffer to hold unique PixelPatch's//
	unsigned long long* PixelPatchUnique;
	cudaMalloc((void**)&PixelPatchUnique, PixelPatchNum * sizeof(unsigned long long));
	thrust::device_ptr<unsigned long long> ppu = thrust::device_pointer_cast(PixelPatchUnique);


	//set pcount//
	thrust::reduce_by_key(fpp, fpp + FragsNum, fo, ppu, ppc);

	cudaFree(FragOnes);
	cudaFree(PixelPatchUnique);

	//set poffset//
	thrust::device_ptr<int> ppo = thrust::device_pointer_cast(xtfoffset);
	thrust::exclusive_scan(ppc, ppc + PixelPatchNum, ppo);

}


__device__
int GetPixelFromPixelPatchKey(unsigned long long pixelpatch)
{
	unsigned long long shifted;
	shifted = pixelpatch >> 32;
	int result = shifted;
	return result;
}

__device__
unsigned long long GenerateDepthPixelKey(float depth, int pixel)
{
	unsigned long long result = pixel;
	result = result << 32;

	//unsigned long long result=0;

	const float lineParameter = depth;
	//uint converted_key = *((uint *)&lineParameter);
	unsigned int converted_key = *((unsigned int*)&lineParameter);
	const unsigned int mask = ((converted_key & 0x80000000) ? 0xffffffff : 0x80000000);
	converted_key ^= mask;

	result |= (unsigned long long)(converted_key);

	return result;

}

//difference between PileKernelWithStart and PileKernelWithDepth is which value is used in the key which will be sorted later//

__global__
//void PileKernelWithStart(int ppnum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, unsigned long long* pkeyBig)
void PileKernelWithStart(int ppnum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, int* ptopBig, unsigned long long* pkeyBig)
{
	int pp = blockIdx.x * blockDim.x + threadIdx.x;

	if (pp < ppnum)
	{
		int count = xtfcount[pp];//get count//
		int offset = xtfoffset[pp];//get offset//

		//get pixel//
		unsigned long long pixelpatch = FragPixelPatch[offset];
		int pixel = GetPixelFromPixelPatchKey(pixelpatch);
		

		//int pileIndex = 0;
		int pileIndex = offset;
		float currstart, currend; int currvertex, currdist;
		float wstartSum, wendSum, wdepthSum, wSum, avgDepth, avgStart, avgEnd; float e;
		for (int index = offset; index < offset + count; index++)
		{
			//read start and end
			float start = FragDepth[index];
			float rad = FragRad[index];
			float end = start + 2 * FragRad[index];
			int vertex = FragVertex[index];
			int dist = FragDist[index];
			//get pixel too?



			e = expf(-dist * dist);
			if (e < 0.000001) e = 0.000001;

			if (index == offset) //if newpile, set currstart and currend
			{
				currstart = start;
				currend = end;
				currvertex = vertex;
				currdist = dist;

				wstartSum = e * start;
				wendSum = e * end;
				wSum = e;
			}
			else
			{
				if (start < currend) //if start less than currend, then same pile
				{
					if (end > currend)//if end bigger than currend, update currend
						currend = end;

					//vertex thing
					if (dist < currdist)
					{
						currdist = dist;
						currvertex = vertex;
					}

					wstartSum += e * start;
					wendSum += e * end;
					wSum += e;
				}
				else //if start bigger than currend, then save currpile, and make a new pile (with new currstart and currend)
				{
					//save current pile at pileIndex
					avgStart = wstartSum / wSum;
					avgEnd = wendSum / wSum;
					avgDepth = (avgStart + avgEnd) / 2;
					pstartBig[pileIndex] = avgStart;
					pdepthBig[pileIndex] = avgDepth;
					pvertexBig[pileIndex] = currvertex;
					ppixelBig[pileIndex] = pixel;
					ptopBig[pileIndex] = 0;
					pkeyBig[pileIndex] = GenerateDepthPixelKey(currstart, pixel);
					pileIndex++;

					currstart = start;
					currend = end;
					currvertex = vertex;
					currdist = dist;
					wstartSum = e * start;
					wendSum = e * end;
					wSum = e;
				}
			}
			
		}

		//save last pile at pileIndex
		avgStart = wstartSum / wSum;
		avgEnd = wendSum / wSum;
		avgDepth = (avgStart + avgEnd) / 2;
		pstartBig[pileIndex] = avgStart;
		pdepthBig[pileIndex] = avgDepth;
		pvertexBig[pileIndex] = currvertex;
		ppixelBig[pileIndex] = pixel;
		ptopBig[pileIndex] = 0;
		pkeyBig[pileIndex] = GenerateDepthPixelKey(currstart, pixel);
		
		
	}

}

__global__
//void PileKernelWithDepth(int ppnum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, unsigned long long* pkeyBig)
void PileKernelWithDepth(int ppnum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, int* ptopBig, unsigned long long* pkeyBig)
{
	int pp = blockIdx.x * blockDim.x + threadIdx.x;

	if (pp < ppnum)
	{
		int count = xtfcount[pp];//get count//
		int offset = xtfoffset[pp];//get offset//

		//get pixel//
		unsigned long long pixelpatch = FragPixelPatch[offset];
		int pixel = GetPixelFromPixelPatchKey(pixelpatch);


		//int pileIndex = 0;
		int pileIndex = offset;
		float currstart, currend; int currvertex, currdist;
		float wstartSum, wendSum, wdepthSum, wSum, avgDepth, avgStart, avgEnd; float e;
		for (int index = offset; index < offset + count; index++)
		{
			//read start and end
			float start = FragDepth[index];
			float rad = FragRad[index];
			float end = start + 2 * FragRad[index];
			int vertex = FragVertex[index];
			int dist = FragDist[index];
			//get pixel too?



			e = expf(-dist * dist);
			if (e < 0.000001) e = 0.000001;

			if (index == offset) //if newpile, set currstart and currend
			{
				currstart = start;
				currend = end;
				currvertex = vertex;
				currdist = dist;

				wstartSum = e * start;
				wendSum = e * end;
				wSum = e;
			}
			else
			{
				if (start < currend) //if start less than currend, then same pile
				{
					if (end > currend)//if end bigger than currend, update currend
						currend = end;

					//vertex thing
					if (dist < currdist)
					{
						currdist = dist;
						currvertex = vertex;
					}

					wstartSum += e * start;
					wendSum += e * end;
					wSum += e;
				}
				else //if start bigger than currend, then save currpile, and make a new pile (with new currstart and currend)
				{
					//save current pile at pileIndex
					avgStart = wstartSum / wSum;
					avgEnd = wendSum / wSum;
					avgDepth = (avgStart + avgEnd) / 2;
					pstartBig[pileIndex] = avgStart;
					pdepthBig[pileIndex] = avgDepth;
					pvertexBig[pileIndex] = currvertex;
					ppixelBig[pileIndex] = pixel;
					ptopBig[pileIndex] = 0;
					pkeyBig[pileIndex] = GenerateDepthPixelKey(avgDepth, pixel);
					pileIndex++;

					currstart = start;
					currend = end;
					currvertex = vertex;
					currdist = dist;
					wstartSum = e * start;
					wendSum = e * end;
					wSum = e;
				}
			}

		}

		//save last pile at pileIndex
		avgStart = wstartSum / wSum;
		avgEnd = wendSum / wSum;
		avgDepth = (avgStart + avgEnd) / 2;
		pstartBig[pileIndex] = avgStart;
		pdepthBig[pileIndex] = avgDepth;
		pvertexBig[pileIndex] = currvertex;
		ppixelBig[pileIndex] = pixel;
		ptopBig[pileIndex] = 0;
		pkeyBig[pileIndex] = GenerateDepthPixelKey(avgDepth, pixel);


	}

}

//void PileCuda(PileSampleType SampleValue, int PixelPatchNum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, unsigned long long* pkeyBig)
void PileCuda(PileSampleType SampleValue, int PixelPatchNum, int* xtfcount, int* xtfoffset, float* FragDepth, float* FragRad, int* FragVertex, int* FragDist, unsigned long long* FragPixelPatch, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, int* ptopBig, unsigned long long* pkeyBig)
{
	if(SampleValue==START)
		//PileKernelWithStart << <PixelPatchNum / 256 + 1, 256 >> > (PixelPatchNum, xtfcount, xtfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch, pstartBig, pdepthBig, pvertexBig, ppixelBig, pkeyBig);
		PileKernelWithStart << <PixelPatchNum / 256 + 1, 256 >> > (PixelPatchNum, xtfcount, xtfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch, pstartBig, pdepthBig, pvertexBig, ppixelBig, ptopBig, pkeyBig);
	else 
		//PileKernelWithDepth << <PixelPatchNum / 256 + 1, 256 >> > (PixelPatchNum, xtfcount, xtfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch, pstartBig, pdepthBig, pvertexBig, ppixelBig, pkeyBig);
		PileKernelWithDepth << <PixelPatchNum / 256 + 1, 256 >> > (PixelPatchNum, xtfcount, xtfoffset, FragDepth, FragRad, FragVertex, FragDist, FragPixelPatch, pstartBig, pdepthBig, pvertexBig, ppixelBig, ptopBig, pkeyBig);
}

__global__
void FindTopPilesKernel(int FragsNum, int* pvertexBig, int* ppixelBig, int* ptopBig, int* vptc)
{

	int pl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pl < FragsNum)
	{
		if (ptopBig[pl] != -1)
		{
			if (pl == 0)
				ptopBig[pl] = 1;
			else
			{
				int pixel = ppixelBig[pl];
				int prevpixel = ppixelBig[pl - 1];
				if (pixel != prevpixel)
					ptopBig[pl] = 1;
				else
				{
					int vertex = pvertexBig[pl];
					int prevvertex = pvertexBig[pl - 1];
					int patch = vptc[vertex];
					int prevpatch = vptc[prevvertex];
					if (patch != prevpatch)
						ptopBig[pl] = 1;
				}
			}
		}
	}

}

void FindTopPilesCuda(int FragsNum, int* pvertexBig, int* ppixelBig, int* ptopBig, int* vptc)
{
	FindTopPilesKernel << < FragsNum / 256 + 1, 256 >> > (FragsNum, pvertexBig, ppixelBig, ptopBig, vptc);
}

struct is_nonnegf
{
	__host__ __device__
		bool operator()(const float x)
	{
		return (x >= 0);
	}
};

struct is_nonnegi
{
	__host__ __device__
		bool operator()(const int x)
	{
		return (x >= 0);
	}
};

struct is_nonmaxull
{
	__host__ __device__
		bool operator()(const unsigned long long x)
	{
		return (x != 0xFFFFFFFFFFFFFFFF);
	}
};

int CountPilesCuda(int FragsNum, float* pstartBig)
{
	thrust::device_ptr<float> ps = thrust::device_pointer_cast(pstartBig);
	int PilesNum = thrust::count_if(ps, ps + FragsNum, is_nonnegf());

	return PilesNum;
}

//void CleanPilesCuda(int FragsNum, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, unsigned long long* pkeyBig, float* pstart, float* pdepth, int* pvertex, int* ppixel, unsigned long long* pkey)
void CleanPilesCuda(int FragsNum, float* pstartBig, float* pdepthBig, int* pvertexBig, int* ppixelBig, int* ptopBig, unsigned long long* pkeyBig, float* pstart, float* pdepth, int* pvertex, int* ppixel, int* ptop, unsigned long long* pkey)
{
	thrust::device_ptr<float> psB = thrust::device_pointer_cast(pstartBig);
	thrust::device_ptr<float> ps = thrust::device_pointer_cast(pstart);
	thrust::copy_if(psB, psB + FragsNum, ps, is_nonnegf());

	thrust::device_ptr<float> pdB = thrust::device_pointer_cast(pdepthBig);
	thrust::device_ptr<float> pd = thrust::device_pointer_cast(pdepth);
	thrust::copy_if(pdB, pdB + FragsNum, pd, is_nonnegf());

	thrust::device_ptr<int> pvB = thrust::device_pointer_cast(pvertexBig);
	thrust::device_ptr<int> pv = thrust::device_pointer_cast(pvertex);
	thrust::copy_if(pvB, pvB + FragsNum, pv, is_nonnegi());

	thrust::device_ptr<int> ppB = thrust::device_pointer_cast(ppixelBig);
	thrust::device_ptr<int> pp = thrust::device_pointer_cast(ppixel);
	thrust::copy_if(ppB, ppB + FragsNum, pp, is_nonnegi());

	thrust::device_ptr<int> ptB = thrust::device_pointer_cast(ptopBig);
	thrust::device_ptr<int> pt = thrust::device_pointer_cast(ptop);
	thrust::copy_if(ptB, ptB + FragsNum, pt, is_nonnegi());

	thrust::device_ptr<unsigned long long> pkB = thrust::device_pointer_cast(pkeyBig);
	thrust::device_ptr<unsigned long long> pk = thrust::device_pointer_cast(pkey);
	thrust::copy_if(pkB, pkB + FragsNum, pk, is_nonmaxull());

	
}

//void SortPilesCuda(int PilesNum, float* pstart, float* pdepth, int* pvertex, int* ppixel, unsigned long long* pkey)
void SortPilesCuda(int PilesNum, float* pstart, float* pdepth, int* pvertex, int* ppixel, int* ptop, unsigned long long* pkey)
{

	//device pointers//
	thrust::device_ptr<float> ps = thrust::device_pointer_cast(pstart);
	thrust::device_ptr<float> pd = thrust::device_pointer_cast(pdepth);
	thrust::device_ptr<int> pv = thrust::device_pointer_cast(pvertex);
	thrust::device_ptr<int> pp = thrust::device_pointer_cast(ppixel);
	thrust::device_ptr<int> pt = thrust::device_pointer_cast(ptop);
	thrust::device_ptr<unsigned long long> pk = thrust::device_pointer_cast(pkey);

	unsigned int* PileIndex;
	cudaMalloc((void**)&PileIndex, PilesNum * sizeof(unsigned int));
	thrust::device_ptr<unsigned int> pi = thrust::device_pointer_cast(PileIndex);
	thrust::sequence(pi, pi + PilesNum, 0);
	//cudaDeviceSynchronize();


	thrust::sort_by_key(pk, pk + PilesNum, pi);
	//cudaDeviceSynchronize();
	

	//tmp buffers for thrust::gather//
	float* pstartTmp;
	float* pdepthTmp;
	int* pvertexTmp;
	int* ppixelTmp;
	int* ptopTmp;
	cudaMalloc((void**)&pstartTmp, PilesNum * sizeof(float));
	cudaMalloc((void**)&pdepthTmp, PilesNum * sizeof(float));
	cudaMalloc((void**)&pvertexTmp, PilesNum * sizeof(int));
	cudaMalloc((void**)&ppixelTmp, PilesNum * sizeof(int));
	cudaMalloc((void**)&ptopTmp, PilesNum * sizeof(int));
	thrust::device_ptr<float> pst = thrust::device_pointer_cast(pstartTmp);
	thrust::device_ptr<float> pdt = thrust::device_pointer_cast(pdepthTmp);
	thrust::device_ptr<int> pvt = thrust::device_pointer_cast(pvertexTmp);
	thrust::device_ptr<int> ppt = thrust::device_pointer_cast(ppixelTmp);
	thrust::device_ptr<int> ptt = thrust::device_pointer_cast(ptopTmp);
	//cudaDeviceSynchronize();

	//change all other arrays based on the sorted index//
	thrust::gather(pi, pi + PilesNum, ps, pst);
	thrust::gather(pi, pi + PilesNum, pd, pdt);
	thrust::gather(pi, pi + PilesNum, pv, pvt);
	thrust::gather(pi, pi + PilesNum, pp, ppt);
	thrust::gather(pi, pi + PilesNum, pt, ptt);
	//cudaDeviceSynchronize();
	cudaMemcpy(pstart, pstartTmp, PilesNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pdepth, pdepthTmp, PilesNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pvertex, pvertexTmp, PilesNum * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(ppixel, ppixelTmp, PilesNum * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(ptop, ptopTmp, PilesNum * sizeof(int), cudaMemcpyDeviceToDevice);
	//cudaDeviceSynchronize();

	cudaFree(pstartTmp);
	cudaFree(pdepthTmp);
	cudaFree(pvertexTmp);
	cudaFree(ppixelTmp);
	cudaFree(ptopTmp);

}

//use a template?
void FillAllWithValue(int* arr, int sz, int val)
{
	
	thrust::device_ptr<int> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(unsigned int* arr, int sz, unsigned int val)
{

	thrust::device_ptr<unsigned int> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(float* arr, int sz, float val)
{

	thrust::device_ptr<float> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(bool* arr, int sz, bool val)
{

	thrust::device_ptr<bool> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val)
{
	thrust::device_ptr<unsigned long long> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);
}

__global__
void CountOcclusionsKernel(int FPixelsNum, int* xpcount, int* xpoffset, float* pstart, float* pdepth, int* ptop, int* xocount)
{
	int pxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pxl < FPixelsNum)
	{
		int piles_start = xpoffset[pxl];
		int piles_end = piles_start + xpcount[pxl] - 1;

		if (piles_start < piles_end)
		{
			int count = 0;
			for (int pile1 = piles_start; pile1 <= piles_end; pile1++)
			{
				if (ptop[pile1] != 1)
					continue;
				float rad1 = pdepth[pile1] - pstart[pile1];
				float start1 = pdepth[pile1] - 1.5 * rad1;
				float end1 = pdepth[pile1] + 1.5 * rad1;
				for (int pile2 = pile1 + 1; pile2 <= piles_end; pile2++)
				{
					if (ptop[pile2] != 1)
						continue;

					float rad2 = pdepth[pile2] - pstart[pile2];
					float start2 = pdepth[pile2] - 1.5 * rad2;
					float end2 = pdepth[pile2] + 1.5 * rad2;

					//check intersection. continue if yes//
					if (((start1 <= start2) && (start2 <= end1)) || ((start1 <= end2) && (end2 <= end1)) || ((start2 <= start1) && (end1 <= end2)))
						continue;


					//increment counter//
					count++;
				}

			}
			xocount[pxl] = count; //set occlusions per pixel counter//
		}
	}
}

void CountOcclusionsCuda(int FPixelsNum, int* xpcount, int* xpoffset, float* pstart, float* pdepth, int* ptop, int* xocount)
{
	CountOcclusionsKernel << < FPixelsNum / 256 + 1, 256 >> > (FPixelsNum, xpcount, xpoffset, pstart, pdepth, ptop, xocount);
}

__device__
unsigned long long GenerateOcclusionPair(int occluder, int occludee)
{

	unsigned long long result = occluder;
	result = result << 32;


	const int lineParameter = occludee;
	unsigned int converted_key = *((unsigned int*)&lineParameter);

	result |= (unsigned long long)(converted_key);

	return result;

}

__global__
void FindOcclusionsKernel(int PixelsNum, int* xpcount, int* xpoffset, float* pstart, float* pdepth, int* pvertex, int* ptop, int* vptc, int* xooffset, unsigned long long* occpair)
{
	int pxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pxl < PixelsNum)
	{
		int piles_start = xpoffset[pxl];
		int piles_end = piles_start + xpcount[pxl] - 1;

		if (piles_start < piles_end)
		{
			int index = xooffset[pxl];

			for (int pile1 = piles_start; pile1 <= piles_end; pile1++)
			{
				if (ptop[pile1] != 1)
					continue;
				float rad1 = pdepth[pile1] - pstart[pile1];
				float start1 = pdepth[pile1] - 1.5 * rad1;
				float end1 = pdepth[pile1] + 1.5 * rad1;
				for (int pile2 = pile1 + 1; pile2 <= piles_end; pile2++)
				{
					if (ptop[pile2] != 1)
						continue;

					float rad2 = pdepth[pile2] - pstart[pile2];
					float start2 = pdepth[pile2] - 1.5 * rad2;
					float end2 = pdepth[pile2] + 1.5 * rad2;

					//check intersection. continue if yes//
					if (((start1 <= start2) && (start2 <= end1)) || ((start1 <= end2) && (end2 <= end1)) || ((start2 <= start1) && (end1 <= end2)))
						continue;


					//increment in pixel counter//
					//atomicAdd(&xocount[pxl], 1);

					int vertex1 = pvertex[pile1];
					int vertex2 = pvertex[pile2];
					int patch1 = vptc[vertex1];
					int patch2 = vptc[vertex2];

					unsigned long long pair = GenerateOcclusionPair(patch1, patch2);
					occpair[index] = pair;

					index++;

				}

			}
		}
	}
}

void FindOcclusionsCuda(int PixelsNum, int* xpcount, int* xpoffset, float* pstart, float* pdepth, int* pvertex, int* ptop, int* vptc, int* xooffset, unsigned long long* occpair)
{

	FindOcclusionsKernel << < PixelsNum / 256 + 1, 256 >> > (PixelsNum, xpcount, xpoffset, pstart, pdepth, pvertex, ptop, vptc, xooffset, occpair);

}

int SetPileCountBigCuda(int PilesNum, int PixelsNum, int* ppixel, int* xpcountBig)
{
	//reduce_by_key input//

	thrust::device_ptr<int> pp = thrust::device_pointer_cast(ppixel);


	int* PilesOnes;			//a buffer of ones//
	cudaMalloc((void**)&PilesOnes, PilesNum * sizeof(int));
	thrust::device_ptr<int> pos = thrust::device_pointer_cast(PilesOnes);
	thrust::fill(pos, pos + PilesNum, 1);


	//reduce_by_key output//

	int* filledpixels;		//empty buffer for indeces of pixels with piles. not important//
	cudaMalloc((void**)&filledpixels, PixelsNum * sizeof(int));
	thrust::device_ptr<int> fp = thrust::device_pointer_cast(filledpixels);


	thrust::device_ptr<int> pcb = thrust::device_pointer_cast(xpcountBig);

	//reduce_by_key//

	thrust::reduce_by_key(pp, pp + PilesNum, pos, fp, pcb);


	//get number of filled pixels//
	int FPixelsNum = thrust::count_if(pcb, pcb + PixelsNum, is_nonnegi());

	//delete fp//
	cudaFree(filledpixels);
	cudaFree(PilesOnes);

	return FPixelsNum;

}

void SetPileCountAndOffsetCuda(int PixelsNum, int FPixelsNum, int* xpcountBig, int* xpcount, int* xpoffset)
{
	thrust::device_ptr<int> pcb = thrust::device_pointer_cast(xpcountBig);
	thrust::device_ptr<int> pc = thrust::device_pointer_cast(xpcount);
	thrust::device_ptr<int> po = thrust::device_pointer_cast(xpoffset);

	//fill pcount//
	thrust::copy_if(pcb, pcb + PixelsNum, pc, is_nonnegf());

	//fill offset//
	thrust::exclusive_scan(pc, pc + FPixelsNum, po);


}


int GetOcclusionsNumCuda(int FPixelsNum, int* xocount)
{
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xocount);

	//get count of xfcount//
	int OcclusionsNum = thrust::reduce(c, c + FPixelsNum, (int)0, thrust::plus<int>());
	//cudaDeviceSynchronize();

	return OcclusionsNum;
}

void SetOcclusionOffsetCuda(int FPixelsNum, int* xocount, int* xooffset)
{
	//find num of occlusions and return

	thrust::device_ptr<int> o = thrust::device_pointer_cast(xooffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xocount);

	//call thrust function
	thrust::exclusive_scan(c, c + FPixelsNum, o);
	//cudaDeviceSynchronize();
}


void SetOccpairCompactAndCountBigCuda(int OcclusionsNum, unsigned long long* occpair, unsigned long long* occpairCompactBig, int* occpairCompactCountBig)
{

	thrust::device_ptr<unsigned long long> op = thrust::device_pointer_cast(occpair);
	thrust::sort(op, op + OcclusionsNum);

	int* OccsOnes;			//a buffer of ones//
	cudaMalloc((void**)&OccsOnes, OcclusionsNum * sizeof(int));
	thrust::device_ptr<int> oos = thrust::device_pointer_cast(OccsOnes);
	thrust::fill(oos, oos + OcclusionsNum, 1);

	thrust::device_ptr<unsigned long long> opcb = thrust::device_pointer_cast(occpairCompactBig);
	thrust::device_ptr<int> opccb = thrust::device_pointer_cast(occpairCompactCountBig);

	thrust::reduce_by_key(op, op + OcclusionsNum, oos, opcb, opccb);

	cudaFree(OccsOnes);
}

struct is_posi
{
	__host__ __device__
		bool operator()(const int x)
	{
		return (x > 0);
	}
};

struct is_posull
{
	__host__ __device__
		bool operator()(const unsigned long long x)
	{
		return (x > 0);
	}
};

int GetCompactOcclusionsNumCuda(int OcclusionsNum, int* occpairCompactCountBig)
{
	thrust::device_ptr<int> opccb = thrust::device_pointer_cast(occpairCompactCountBig);

	//get new number of occlusions//
	int newOcclusionsNum = thrust::count_if(opccb, opccb + OcclusionsNum, is_posi());


	return newOcclusionsNum;

}

void SetOccpairCompactAndCountCuda(int OcclusionsNum, unsigned long long* occpairCompactBig, int* occpairCompactCountBig, unsigned long long* occpairCompact, int* occpairCompactCount)
{
	thrust::device_ptr<unsigned long long> opcb = thrust::device_pointer_cast(occpairCompactBig);
	thrust::device_ptr<int> opccb = thrust::device_pointer_cast(occpairCompactCountBig);
	thrust::device_ptr<unsigned long long> opc = thrust::device_pointer_cast(occpairCompact);
	thrust::device_ptr<int> opcc = thrust::device_pointer_cast(occpairCompactCount);

	//fill pairs and pairs count//
	thrust::copy_if(opccb, opccb + OcclusionsNum, opcc, is_posi());
	thrust::copy_if(opcb, opcb + OcclusionsNum, opc, is_posull());

}
