
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE	32
#define BLOCK_SIZE2	BLOCK_SIZE
#define WR		1
#define HR		8192
#define WB		8192
#define HB		WR
#define WA		WB
#define HA		HR

void matrixGPU(float* _MAX, float* _MIN, float *_C, const float *_A, const float *_B, int _wa, int _ha, int _wb);
void randomInit(float* _data, int _size);

//C = A*B
void matrixMulCPU(float* _C, const float *_A, const float *_B, int _wa, int _ha, int _wb)
{
	float sum = 0;
	for (int i = 0; i < _ha; ++i)
	{
		for (int j = 0; j < _wb; ++j)
		{
			sum = 0;
			for (int k = 0; k < _wa; ++k)
			{
				sum += (float)_A[i*_wa + k] * (float)_B[k*_wb + j];
			}
			_C[i*_wb + j] = (float)sum;
		}
	}
}

void matrixTransposeCPU(float* odata, const float *idata, int width, int height)
{
	int i, j;
	for (i = 0; i < width; ++i){
		for (j = 0; j < height; ++j){
			odata[width*i + j] = idata[i*height + j];
		}
	}
}

__global__ void matrixMul_kernel(float* _C, const float* _A, const float *_B, int _wa, int _wb)
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

// 矩阵乘
__global__ void matrixMUL_kernel_shared(float* _C, const float* _A, const float *_B, int _wa, int _wb)
{
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
		__shared__ float As[BLOCK_SIZE2][BLOCK_SIZE2];
		__shared__ float Bs[BLOCK_SIZE2][BLOCK_SIZE2];
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

// 平方和
__global__ void reduction_kernel(float* _odata, const float* _idata, const unsigned int _size)
{
	__shared__ float partialSum[BLOCK_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	float sum = (i < _size) ? _idata[i] * _idata[i] : 0;
	if (i + blockDim.x < _size)
		sum += _idata[i + blockDim.x] * _idata[i + blockDim.x];
	partialSum[t] = sum;

	unsigned int stride;
	for (stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (t < stride)
			//partialSum[t] += partialSum[t + stride];
			partialSum[t] = sum = sum + partialSum[t + stride] * partialSum[t + stride];
	}

	if (t == 0)
		_odata[blockIdx.x] = sqrt(partialSum[0]);
}

// 除法 + 查找最大和最小（合并规约）
__global__ void reduction2_kernel(float* _omax, float* _omin, const float* _idata, const float _sum, const unsigned int _size)
{
	__shared__ float partialMAX[BLOCK_SIZE];
	__shared__ float partialMIN[BLOCK_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	//partialSum[t] = (i < _size) ? _idata[i] : 0;
	float tmp = (i < _size) ? _idata[i] / _sum : 0;
	partialMAX[t] = partialMIN[t] = tmp;
	if (i + blockDim.x < _size){
		tmp = _idata[i + blockDim.x] / _sum;
		if (tmp > partialMAX[t]) partialMAX[t] = tmp;
		if (tmp < partialMIN[t]) partialMIN[t] = tmp;
	}

	unsigned int stride;
	for (stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (t < stride){
			tmp = _idata[t + stride] / _sum;
			if (tmp > partialMAX[t]) partialMAX[t] = tmp;
			if (tmp < partialMIN[t]) partialMIN[t] = tmp;
		}
	}

	if (t == 0)
	{
		_omax[blockIdx.x] = partialMAX[0];
		_omin[blockIdx.x] = partialMIN[0];
	}
}

int main()
{
	srand((unsigned)time(NULL));

	float* cpu_R = (float*)malloc(sizeof(float)*WR*HR);
	float* cpu_B = (float*)malloc(sizeof(float)*WB*HB);
	int i;

	randomInit(cpu_R, WR*HR);

	float* cpu_A = (float*)malloc(sizeof(float)*WA*HA);
	memset(cpu_A, 0, sizeof(float)*WA*HA);

	//----- w/ CPU
	clock_t tick1 = clock();
	//matrixTransposeCPU(cpu_B, cpu_R, WR, HR);
	matrixMulCPU(cpu_A, cpu_R, cpu_R, WR, HR, WB);

	float sum = 0, max = 0, min = 1, tmp = 0;
	for (i=0; i<HR; ++i){
		sum += (cpu_R[i] * cpu_R[i]);
	}

	sum = sqrt(sum);
	for (i = 0; i < WA * HA; ++i){
		tmp = cpu_A[i] / sum;
		max = (tmp > max) ? tmp : max;
		min = (tmp < min) ? tmp : min;
	}

	double time = (double)((clock() - tick1) / (double)CLOCKS_PER_SEC);
	printf("max(A): %f, min(A): %f\n", max, min);
	printf("CPU use Time: %f (ms)\n", time * 1000);
	//-----

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float max2, min2;
	matrixGPU(&max2, &min2, cpu_A, cpu_R, cpu_R, WR, HR, WB);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU use time: %f (ms), Block Size: %d \n", elapsedTime, BLOCK_SIZE);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("%f, %f, %f, %f", max, min, max2, min2);
	if ((max == max2) && (min == min2))
	{
		printf("Accept\n");
	}
	else
	{
		printf("Worng Answer\n");
	}

	free(cpu_A);
	free(cpu_B);
	free(cpu_R);

	cudaDeviceReset();
	return 0;
}

void matrixGPU(float* _MAX, float* _MIN, float *_C, const float *_A, const float *_B, int _wa, int _ha, int _wb)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	float *dev_sum = 0;
	float *dev_max = 0;
	float *dev_min = 0;

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_a, _wa * _ha * sizeof(float));
	cudaMalloc((void**)&dev_b, _wb * _wa * sizeof(float));
	cudaMalloc((void**)&dev_c, _wb * _ha * sizeof(float));
	cudaMalloc((void**)&dev_sum, sizeof(float));
	cudaMalloc((void**)&dev_max, sizeof(float));
	cudaMalloc((void**)&dev_min, sizeof(float));


	cudaMemcpy(dev_a, _A, _wa * _ha * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, _B, _wb * _wa * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(WA / BLOCK_SIZE, HA / BLOCK_SIZE);
	matrixMUL_kernel_shared << <blocks, threads >> >(dev_c, dev_a, dev_b, _wa, _wb);
	cudaThreadSynchronize();

	reduction_kernel << <blocks, threads >> >(dev_sum, dev_c, (WA * HA));
	cudaThreadSynchronize();

	reduction2_kernel << <blocks, threads >> >(dev_max, dev_min, dev_c, *dev_sum, (WA * HA));
	cudaThreadSynchronize();

	cudaMemcpy(_MAX, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(_MIN, dev_min, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_sum);
	cudaFree(dev_max);
	cudaFree(dev_min);

}

void randomInit(float* _data, int _size)
{
	for (int i = 0; i < _size; ++i)
	{
		_data[i] = rand() / (float)RAND_MAX;
	}
}