/*
 * =====================================================================================
 *
 *       Filename:  predefine.h
 *
 *    Description:	predefined structure and union 
 *
 *        Version:  1.0
 *        Created:  07/29/2024 01:14:46 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon (MK), myungkuk.yoon@ewha.ac.kr
 *   Organization:  Department of Computer Science and Engineering
 *
 * =====================================================================================
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cstring>
#include <assert.h>
#include <math.h>

const unsigned char IMG_SIZE = 28;
const unsigned char FILENAME_SIZE = 100;

typedef union chars_to_uint{
	char c_array[4];
	unsigned int uint;
} c2u;

typedef union uint_to_float{
	unsigned int uint;
	float fp;
} u2f;

typedef union singleD_to_multiD{
	float oneD[IMG_SIZE * IMG_SIZE];
	float twoD[IMG_SIZE][IMG_SIZE];
} sd2md;

typedef struct mnist_data{
	unsigned char data[IMG_SIZE][IMG_SIZE];
	unsigned char label;

	sd2md nor_data;
	//using bit ==> 9 ==> 0000 0010 0000 0000
	unsigned short bit_label;
} m_data;

typedef struct full_layer{
	//for CPU
	u2f *weights;
	u2f *bias;
	
	float *result;

	//for GPU 
	float *d_weights;
	float *d_bias;

	float *d_result;

	unsigned num_weight[2];
	unsigned num_bias[2];

	unsigned weightSize;
	unsigned biasSize;
	unsigned resultSize;

	bool perf_act;
} fLayer;

