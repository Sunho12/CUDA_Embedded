/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  image blur cpu code
 *
 *        Version:  1.0
 *        Created:  07/14/2021 10:41:21 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@ewha.ac.kr
 *   Organization:  EWHA Womans Unversity
 *
 * =====================================================================================
 */

#include<iostream>
#include "ppm.h"
#include "clockMeasure.h"

#define BLUR_SIZE 5

using namespace std;

const int MAX_ITER = 10;

void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){

	for(int y=0; y < h; y++) {
		for (int x=0; x < w; x++) {
			int r = 0, g = 0, b = 0;
			int count = 0;

			for (int ky= -BLUR_SIZE; ky <= BLUR_SIZE; ky++ ){
				for (int kx = -BLUR_SIZE; kx <= BLUR_SIZE; kx++){
					int ny = y + ky;
					int nx = x + kx;

					if(nx >= 0 && nx < w && ny >= 0 && ny < h){
						int index = (ny * w + nx) * 3;
						r += inArray[index];
						b += inArray[index + 1];
						g += inArray[index + 2];
						count++;
					}
				}
			}
		

			if (count > 0) {
                int index = (y * w + x) * 3;
                outArray[index] = r / count;
                outArray[index + 1] = g / count;
                outArray[index + 2] = b / count;
            }
		}
	}
}

int main(){
	int w, h;
	unsigned char *h_imageArray;
	unsigned char *h_outImageArray;

	//This function will load the R/G/B values from a PPM file into an array and return the width (w) and height (h).
	ppmLoad("./data/ewha_picture.ppm", &h_imageArray, &w, &h);

	clockMeasure *ckCpu = new clockMeasure("CPU CODE");

	ckCpu->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuCode(h_outImageArray, h_imageArray, w, h); 
		ckCpu->clockPause();
	}
	ckCpu->clockPrint();

	//This function will store the R/G/B values from h_outImageArray into a PPM file.
	ppmSave("ewha_picture_cpu.ppm", h_outImageArray, w, h);

	return 0;
}
