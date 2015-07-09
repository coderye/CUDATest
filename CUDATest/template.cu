
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE	32
#define BLOCK_SIZE2	32//64//16
#define WA		16384
#define HA		4096
#define WB		2048
#define HB		WA
#define WC		WB
#define HC		HA

void matrixMulGPU(float* _C, const float *_A, const float *_B, int _wa, int _ha, int _wb);
void matrixMulGPU2(float* _C, const float *_A, const float *_B, int _wa, int _ha, int _wb);
void randomInit(float* _data, int _size);
bool checkError(const float* _A, const float* _B, int _size);

__global__ void matrix_kernel(float* _C, const float* _A, const float *_B, int _wa, int _wb)
{
	float sum = 0;
	//找出该线程所在的行列
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	//线程Thread(row,col)负责计算C(row,col)
	for (int i = 0; i < _wa; ++i)
	{
		sum += _A[row*_wa + i] * _B[i*_wb + col];
	}
	_C[row*_wb + col] = sum;
}

__global__ void matrix_kernel_shared(float* _C, const float* _A, const float *_B, int _wa, int _wb)
{
	__shared__ float As[BLOCK_SIZE2][BLOCK_SIZE2];
	__shared__ float Bs[BLOCK_SIZE2][BLOCK_SIZE2];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//该block要处理的A
	int aBegin = _wa*(by*BLOCK_SIZE2);//A(0,by)
	int aEnd = aBegin + _wa - 1;
	int aStep = BLOCK_SIZE2;//offsetA

	int bBegin = BLOCK_SIZE2*bx;//B(bx,0)
	int bStep = BLOCK_SIZE2*_wb;//offsetB

	float cSub = 0;
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		//每个线程负责一个元素拷贝
		As[ty][tx] = _A[a + _wa*ty + tx];
		Bs[ty][tx] = _B[b + _wb*ty + tx];

		__syncthreads();

		//每个线程负责计算一个子块i 和 子块j的子乘积
		for (int k = 0; k < BLOCK_SIZE2; ++k)
		{
			cSub += As[ty][k] * Bs[k][tx];
		}

		__syncthreads();
	}

	//全局地址，向全局寄存器写回去
	//一个线程负责一个元素，一个block负责一个子块
	int cIndex = (by*BLOCK_SIZE2 + ty)*_wb + (bx*BLOCK_SIZE2 + tx);
	_C[cIndex] = cSub;
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

	float* cpu_C2 = (float*)malloc(sizeof(float)*WC*HC);
	memset(cpu_C2, 0, sizeof(float)*WC*HC);

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

	//----- Matrix Mul w/ GPU Shared Memory
	cudaEventRecord(start, 0);

	matrixMulGPU2(cpu_C, cpu_A, cpu_B, WA, HA, WB);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU use time: %f (ms), w/ shared memory, Block Size: %d\n", elapsedTime, BLOCK_SIZE2);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(cpu_A);
	free(cpu_B);
	free(cpu_C);
	free(cpu_C2);

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

void matrixMulGPU2(float* _C, const float *_A, const float *_B, int _wa, int _ha, int _wb)
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

	dim3 threads(BLOCK_SIZE2, BLOCK_SIZE2);
	dim3 blocks(WC / (BLOCK_SIZE2), HC / (BLOCK_SIZE2));
	matrix_kernel_shared << <blocks, threads >> >(dev_c, dev_a, dev_b, _wa, _wb);

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

bool checkError(const float* _A, const float* _B, int _size)
{
	for (int i = 0; i < _size; ++i)
	{
		if (fabs(_A[i] - _B[i]) > 1.0e-3)
		{
			printf("%f \t %f\n", _A[i], _B[i]);
			return false;
		}
	}
	return true;
}
