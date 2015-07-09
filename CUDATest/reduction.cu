
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE	32
#define WA		16384
#define HA		4096
#define WB		2048
#define HB		WA
#define WC		WB
#define HC		HA

void matrixMulGPU(float* _C, const float *_A, const float *_B, int _wa, int _ha, int _wb);
void randomInit(float* _data, int _size);

__global__ void reduction_kernel(float* _odata, const float* _idata, const unsigned int _size)
{
	__shared__ float partialSum[];
	unsigned int t = threadIdx.x;
	//unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	//partialSum[t] = (i < _size) ? _idata[i] : 0;
	float sum = i < _size) ? _idata[i] : 0;
	if (i + blockDim.x < _size)
		sum += _idata[i + blockDim.x];
	partialSum[t] = sum;
	
	unsigned int stride;
	for (stride = blockDim.x/2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (t < stride)
			//partialSum[t] += partialSum[t + stride];
			partialSum[t] = sum = sum + partialSum[t + stride];
	}
	
	if (t == 0)
		_odata[blockIdx.x] = partialSum[0];
}

int main()
{
	srand((unsigned)time(NULL));

	float* cpu_A = (float*)malloc(sizeof(float)*WA*HA);
	float* cpu_B = (float*)malloc(sizeof(float)*WB*HB);

	randomInit(cpu_A, WA*HA);
	randomInit(cpu_B, WB*HB);

	float* cpu_C = (float*)malloc(sizeof(float)*WC*HC);
	memset(cpu_C, 0, sizeof(float)*WC*HC);

	//----- Matrix Mul w/ GPU
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	matrixMulGPU(cpu_C, cpu_A, cpu_B, WA, HA, WB);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU use time: %f (ms), Block Size: %d \n", elapsedTime, BLOCK_SIZE);
	//-----

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(cpu_A);
	free(cpu_B);
	free(cpu_C);

	cudaDeviceReset();
	return 0;
}

void matrixMulGPU(float* _C, const float *_A, const float *_B, int _wa, int _ha, int _wb)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_a, _wa * _ha * sizeof(float));
	cudaMalloc((void**)&dev_b, _wb * _wa * sizeof(float));
	cudaMalloc((void**)&dev_c, _wb * _ha * sizeof(float));

	cudaMemcpy(dev_a, _A, _wa * _ha * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, _B, _wb * _wa * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(WC / BLOCK_SIZE, HC / BLOCK_SIZE);
	matrix_kernel << <blocks, threads >> >(dev_c, dev_a, dev_b, _wa, _wb);

	cudaThreadSynchronize();
	cudaMemcpy(_C, dev_c, _wb * _ha * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void randomInit(float* _data, int _size)
{
	for (int i = 0; i < _size; ++i)
	{
		_data[i] = rand() / (float)RAND_MAX;
	}
}