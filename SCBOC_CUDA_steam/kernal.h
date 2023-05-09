#ifndef __KERNAL_H_
#define __KERNAL_H_

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "gpssim.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void multiArray2D_Wrapper(short* h_A, short* h_B, short* dev_CCos, short* dev_CSin, short* d_iq_buff, int rows, int cols, short* h_iq_buff);
void GPUMemoryInit(Table* Tb, int ch_num);
void produce_samples_withCuda(Table* Tb, int fs, double* parameters, int* sum, int satnum, float* dev_noise, double* db_para);
void produce_samples_withCuda_Firstsec(Table* Tb, int fs, double* parameters, int* sum, int satnum, float* dev_noise, double* db_para);
void GPUMemroy_delete(Table* Tb);
void Parameter_memory(Table* Tb, channel_t* channel, int fs, double* parameters, int* sum, int satnum, double* db_para);
void quantify_with_cuda(float* dev_iq_buff, float* iq_buff, unsigned char* out_buff, unsigned char* dev_out_buff, int iq_buff_size, float* th);
void navdata_update(Table* Tb);
#endif