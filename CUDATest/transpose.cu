#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_DIM 16

__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

int main()
{
	cudaSetDevice(0);
	
	float *cpu_A;
	//float *cpu_B;
	float *gpu_A;
	float *gpu_B;

	int width = 4096;
	int height = 16384;

	size_t pitch;
	int r, c;

	cpu_A = (float*)malloc(sizeof(float)*width*height);
	cudaMallocPitch((void**)&gpu_A, &pitch, sizeof(float)*width, height);

	printf("The pitch is: %d\n", pitch);
	
	for ( r = 0; r < height; ++r){
		for (c = 0; c < width; ++c){
			cpu_A[r*width+c] = r*c;
		}
	}

	cudaMemcpy2D(gpu_A, pitch, cpu_A, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(gpu_B, pitch, cpu_A, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice);	

	dim3 Dg(1, 2, 1);
	dim3 Db(width, 1, 1);
	transpose<<<Dg, Db, 0>>>(gpu_B, gpu_A, width, height);

	cudaMemcpy2D(cpu_A, sizeof(float)*width, gpu_B, pitch, sizeof(float)*width, height, cudaMemcpyDeviceToHost);

	//print result
	/*
	printf("After transpose, CPU_B DATA:\n");
	for (r = 0; r < height; ++r){
		for (c = 0; c < width; ++c){
			printf("%f/t", cpu_A[r*width+c]);
		}
		printf("\n");
	}
	*/
	
	free(cpu_A);
	cudaFree(gpu_A);
	
	return 0;
}
