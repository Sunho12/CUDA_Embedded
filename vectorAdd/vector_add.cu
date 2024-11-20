/*
 * =====================================================================================
 *
 *       Filename:  vector_add.cu
 *
 *    Description: Vector Addition
 *
 *        Version:  1.0
 *        Created:  07/08/2021 09:54:25 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon 
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

/* 
 * There are five exercises where some code is missing. Please fill in the missing code to complete the vector addition program
 */

//Exercise #05: Change MAX_SIZE to see how the performance changes
const int MAX_SIZE=10000; 
const float MAX_NUM=100.0;
const int MAX_ITER= 100;

#include <iostream>
#include "clockMeasure.h"

float inputA[MAX_SIZE];
float inputB[MAX_SIZE];
float gpuAns[MAX_SIZE];
float cpuAns[MAX_SIZE];

using namespace std;

#define checkCudaError(error) 			\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", \
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 				\
		exit(EXIT_FAILURE);							\
	}

void generateRandomValues(float *array, float max, const int size){
	for(int i = 0; i < size; i++){
		array[i] = float(rand())/float(RAND_MAX) * max;
	}
}

void cpuVectorAddition(float *h_a, float *h_b, float *h_c, const int size){
	for(int i = 0; i < size; i++){
		h_c[i] = h_a[i] + h_b[i];
	}
}

__global__
void gpuVectorAddition(float *d_a, float *d_b, float *d_c, const int size){
	//Exercise #04: Insert vector addition code
	int tld = blockDim.x * blockIdx.x + threadIdx.x;
	if(tld < size){
		d_c[tld] = d_a[tld] + d_b[tld];
	}
}

void checkAnswer(float *h_a, float *d_a, const int size){
	bool isSame = true;
	for(int i = 0; i < size; i++){
		if(h_a[i] != d_a[i]){
			cout<<"-\tERROR: IDX - "<< i << " (" << h_a[i] << " != " << d_a[i] << " )" << endl;
			isSame = false;
		}
	}
	if(isSame)
		printf("All values are same\n");
	else
		printf("Some values are not same\n");
}

//Main Function
int main(){
	srand((unsigned int)time(NULL));

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");
	ckCpu->clockReset();
	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();
	
	//Random
	generateRandomValues(inputA, MAX_NUM, MAX_SIZE);
	generateRandomValues(inputB, MAX_NUM, MAX_SIZE);
	
	//GPU Memory
	float *d_a, *d_b, *d_c;
	int arraySize = MAX_SIZE * sizeof(float);
	cudaError_t err = cudaMalloc((void **) &d_a, arraySize);
	checkCudaError(err);
	//Exercise #01: Allocate memory on GPUs for d_b and d_c
	err = cudaMalloc((void **) &d_b, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c, arraySize);

	err = cudaMemcpy(d_a, inputA, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);
	//Exercise #02: Copy data of inputB to d_b
	err = cudaMemcpy(d_b, inputB, arraySize, cudaMemcpyHostToDevice);

	const int tSize = 256;
	dim3 gridSize(ceil((float)MAX_SIZE/(float)tSize), 1, 1);
	dim3 blockSize(tSize, 1, 1);
	
	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuVectorAddition(inputA, inputB, cpuAns, MAX_SIZE);
		ckCpu->clockPause();
		
		ckGpu->clockResume();
		//Exercise #03: Call Kernel Function
		gpuVectorAddition<<<gridSize, blockSize>>>(d_a, d_b, d_c, MAX_SIZE);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);
	}

	err = cudaMemcpy(gpuAns, d_c, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	checkAnswer(cpuAns, gpuAns, MAX_SIZE);

	ckCpu->clockPrint();
	ckGpu->clockPrint();

	return 0;
}
