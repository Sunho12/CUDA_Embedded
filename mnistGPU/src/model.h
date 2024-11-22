/*
 * =====================================================================================
 *
 *       Filename:  model.h
 *
 *    Description:  To load weight from files and execute inference
 *
 *        Version:  1.0
 *        Created:  07/29/2024 12:41:09 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon (MK), myungkuk.yoon@ewha.ac.kr
 *   Organization:  Department of Computer Science and Engineering
 *
 * =====================================================================================
 */

#pragma once

#include "predefine.h"

#define checkCudaError(error) 			\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", \
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 				\
		exit(EXIT_FAILURE);							\
	}

class model{
	public:
		model(unsigned num_layers);
		~model();

		bool read_weights(const char *file, bool activation = true);
		void copy_weights_into_device(fLayer *layer);
		
		unsigned char perf_forward_exec(m_data *img);
		unsigned char perf_forward_exec(float *input);

		unsigned char perf_forward_exec_on_device(m_data *img);
		unsigned char perf_forward_exec_on_device(float *d_input);
		public:
    	unsigned char* perf_forward_exec_on_device_batch(m_data *imgs, unsigned batch_size);


		void perf_fc_exec(fLayer *layer, float *img);
		void perf_act_exec(fLayer *layer);

	private:
		std::vector<fLayer *> m_layers;

		unsigned m_num_layers;
};

