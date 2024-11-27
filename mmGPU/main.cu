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
			   ID:  2076017
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
const int MAX_ITER = 1;

const int NUM_STREAMS = 4;

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
			unsigned int tSum = 0;
			for(int k = 0; k < aColSize; k++){
				tSum += (h_a[i * aColSize + k] * h_b[k * bColSize + j]);
			}
			h_c[i * bColSize + j] = tSum;
		}
	}
}

__global__
void gpuMatrixMul(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(tId < aRowSize * bColSize){
		int rowId = tId / bColSize;
		int colId = tId % bColSize;
		unsigned int tSum = 0;
		for(int i = 0; i < aColSize; i++){
			tSum += (d_a[rowId * aColSize + i] * d_b[i * bColSize + colId]);
		}
		d_c[tId] = tSum;
	}
}

int main(){
	srand((unsigned int)time(NULL));
	generateRandomValues(matrixA, A_H, A_W);
	generateRandomValues(matrixB, B_H, B_W);

	unsigned int *d_a, *d_b, *d_c;
	size_t matrixSizeA = sizeof(unsigned int) * A_H * A_W;
	size_t matrixSizeB = sizeof(unsigned int) * B_H * B_W;
	size_t matrixSizeC = sizeof(unsigned int) * A_H * B_W;

	cudaError_t err = cudaMalloc((void **) &d_a, matrixSizeA);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_b, matrixSizeB);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c, matrixSizeC);
	checkCudaError(err);

	//err = cudaMemcpy(d_a, matrixA, matrixSizeA, cudaMemcpyHostToDevice);
	//checkCudaError(err);
	err = cudaMemcpy(d_b, matrixB, matrixSizeB, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tbSize = 256;
	dim3 gridSize(ceil((float)(A_H * B_W)/(float)tbSize), 1, 1);
	dim3 blockSize(tbSize, 1, 1);
	
	int rowsPerStream = A_H / NUM_STREAMS; // 스트림 별 처리할 행

	cudaStream_t streams[NUM_STREAMS];
	for (int i=0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuMatrixMul(matrixA, matrixB, cpuOut, A_H, A_W, B_H, B_W);
		ckCpu->clockPause();

		ckGpu->clockResume();
		for (int j = 0; j < NUM_STREAMS; j++) {
            int offsetA = j * rowsPerStream * A_W;
            int offsetC = j * rowsPerStream * B_W;

            // 비동기 메모리 복사: 행렬 A의 일부를 GPU로 복사
            cudaMemcpyAsync(&d_a[offsetA], &matrixA[offsetA], rowsPerStream * A_W * sizeof(unsigned int),
                            cudaMemcpyHostToDevice, streams[j]);
            
            // GPU 커널 실행
            gpuMatrixMul<<<gridSize, blockSize, 0, streams[j]>>>(&d_a[offsetA], d_b, &d_c[offsetC], rowsPerStream, A_W, B_W, B_W);

            // 결과 복사: GPU에서 CPU로 결과를 비동기 복사
            cudaMemcpyAsync(&gpuOut[offsetC], &d_c[offsetC], rowsPerStream * B_W * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost, streams[j]);
        }

		for (int i = 0; i< NUM_STREAMS; i++){
			cudaStreamSynchronize(streams[i]);
		}
		ckGpu->clockPause();

	}

	// 스트림 삭제
	for (int i = 0; i<NUM_STREAMS; i++) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	if(compareMatrix(cpuOut, gpuOut, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu->clockPrint();
	}else{
		printf("ERROR: Two Matrices are not same\n");
	}
}
