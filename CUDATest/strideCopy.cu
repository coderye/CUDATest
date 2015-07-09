#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 128*1024*1024
#define BLOCK_SIZE 1024

__global__ void offsetCopy(float *odata, const float *idata, int offset)
{
	int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
	odata[xid] = idata[xid];
}

__global__ void strideCopy(float *odata, const float *idata, int stride)
{
	int xid = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
	odata[xid] = idata[xid];
}

int main()
{
	srand((unsigned)time(NULL));
	
	int i;
	float* cpu_A = (float*)malloc(sizeof(float)*SIZE);
	float* cpu_B = (float*)malloc(sizeof(float)*SIZE);
	memset(cpu_B, 0, sizeof(float)*SIZE);
	for (i = 0; i < SIZE; ++i){
		cpu_A[i] = (float)(rand() / RAND_MAX);
	}

	float *dev_a;
	float *dev_b;

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_a, SIZE * sizeof(float) * 10);
	cudaMalloc((void**)&dev_b, SIZE * sizeof(float) * 10);
	cudaMemcpy(dev_a, cpu_A, SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (i = 1; i <= 10; ++i)
	{
		cudaEventRecord(start, 0);
		
		int blocks = SIZE / BLOCK_SIZE;
		int threads = BLOCK_SIZE;
		strideCopy<<<blocks, threads>>>(dev_b, dev_a, i);

		cudaThreadSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("GPU use time: %f (ms), Stride: %d \n", elapsedTime, i);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
}
