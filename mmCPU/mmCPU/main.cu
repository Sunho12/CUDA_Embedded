/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication
 *
 *        Version:  1.0
 *        Created:  2021/07/30 10:07:38
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Sunho Kwak, latte1210@ewhain.net
			   Id: 2076017
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#include <assert.h>
#include "clockMeasure.h"

#define checkCudaError(error) 					\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", 		\
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 		\
		exit(EXIT_FAILURE);				\
	}

const int A_H = 512;
const int A_W = 512;
const int B_H = A_W;
const int B_W = 512;
const unsigned int MAX_NUM = 100;
const int MAX_ITER = 10;

unsigned int matrixA[A_H * A_W];
unsigned int matrixB[B_H * B_W];
unsigned int cpuOut[A_H * B_W];
unsigned int gpuOut[A_H * B_W];

void generateRandomValues(unsigned int *input, const int rowSize, const int colSize){
	for(int i = 0; i < rowSize; i++){
		for(int j = 0; j < colSize; j++){
			input[i * colSize + j] = (unsigned int) float(rand())/float(RAND_MAX) * MAX_NUM;
		}
	}
}

void printMatrixValue(const unsigned int *input, const int rowSize, const int colSize){
	printf("Print Matrix \n -----------\n");
	for(int i = 0; i < rowSize; i++){
		for(int j = 0; j < colSize; j++){
			printf("%u\t", input[i * colSize + j]);
		}
		printf("\n");
	}
	printf("--------\n");
}

bool compareMatrix(const unsigned int *inputA, const unsigned int *inputB, const int rowSize, const int colSize){
	bool ret = true;
	for(int i = 0; i < rowSize * colSize; i++){
		if(inputA[i] != inputB[i]){
			ret = false;
			break;
		}
	}
	return ret;
}

void cpuMatrixMul(const unsigned int *h_a, const unsigned int *h_b, unsigned int *h_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	for(int i = 0; i < aRowSize; i++){
		for(int j = 0; j < bColSize; j++){
			float tSum = 0.0f;
			for(int k = 0; k < aColSize; k++){
				tSum += (h_a[i * aColSize + k] * h_b[k * bColSize + j]);
			}
			h_c[i * bColSize + j] = tSum;
		}
	}
}

//Write your code
__global__
void gpuMatrixMul(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < aRowSize && col < bColSize) {
		unsigned int sum = 0;
		for (int i = 0; i < aColSize; i++) {
			sum += d_a[row * aColSize + i] * d_b[i * bColSize + col];
		}
		d_c[row * bColSize + col] = sum;
	}
}

int main(){
	//Generating random numbers
	srand((unsigned int)time(NULL));
	generateRandomValues(matrixA, A_H, A_W);
	generateRandomValues(matrixB, B_H, B_W);

	cudaError_t err;

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();

	// GPU 메모리
	unsigned int *d_a, *d_b, *d_c;
	size_t sizeA = A_H * A_W * sizeof(unsigned int);
	size_t sizeB = B_H * B_W * sizeof(unsigned int);
	size_t sizeC = A_H * B_W * sizeof(unsigned int);

	cudaMalloc((void**)&d_a, sizeA);
	cudaMalloc((void**)&d_b, sizeB);
	cudaMalloc((void**)&d_c, sizeC);

	// 데이터 GPU로
	cudaMemcpy(d_a, matrixA, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, matrixB, sizeB, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((B_W + blockDim.x - 1) / blockDim.x, (A_H + blockDim.y - 1) / blockDim.y);


	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuMatrixMul(matrixA, matrixB, cpuOut, A_H, A_W, B_H, B_W);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuMatrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);
	}

	// 결과 CPU로
	cudaMemcpy(gpuOut, d_c, sizeC, cudaMemcpyDeviceToHost);

	if(compareMatrix(cpuOut, gpuOut, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu->clockPrint();
	}else{
		printf("ERROR: Two Matrices are not same\n");
	}

	// GPU 메모리 해제
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
