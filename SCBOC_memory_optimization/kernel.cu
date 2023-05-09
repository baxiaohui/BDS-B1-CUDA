#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "kernal.h"
#include "cuComplex.h"
#include "cuda_texture_types.h"//否则不识别texture
//__constant__  int sinTable512[] = {
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
//};
//
//__constant__ int cosTable512[] = {
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
//};



//#define SIZE 1024
#define BLOCK_SIZES 1024
#define GRID_SIZES 256
#define L 64

texture<int> t_sinTable;//声明纹理参考，用来绑定纹理，其实也就是个纹理标识
texture<int> t_cosTable;
texture<int, 1> t_CAcode;
texture<int, 1>t_NH;
texture<char, 1> t_navbit;

texture<int> t_B1c_data;
texture<int> t_B1c_pilot;
texture<int> t_B1c_secondcode;
texture<int> t_B1c_nav;
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s error code %d\n", cudaGetErrorString(result), result);
		getchar();
		//assert(result == cudaSuccess);
	}
#endif
	return result;
}



__global__ void kernelMultiArray2D(short* ACos, short* ASin, short* B, short* CCos, short* CSin, int rows, int cols, short gain) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * cols;
	int temp = ACos[index];
	ACos[index] = temp * B[index] * CCos[index] * gain;
	ASin[index] = temp * B[index] * CSin[index] * gain;
}

__global__ void kernerSumColumnArray2D(short* ACos, short* ASin, int rows, int cols, short* iq_buff) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;//(13,1*10^6) 256*1024
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * cols;
	if (col < 2)
	{
		printf("col=%d", col);
		printf("row=%d", row);
	}
	if (col < cols && row == 0) {
		int sumCos = 0;
		int sumSin = 0;
		for (int i = 0; i < rows; i++) {
			sumCos += ACos[i * cols + index];
			sumSin += ASin[i * cols + index];
		}
		iq_buff[col * 2] = short((sumCos + 64) >> 7);
		iq_buff[col * 2 + 1] = short((sumSin + 64) >> 7);

	}
}

//multiArray2D_Wrapper(cos_page, sin_page, dev_CCos, dev_CSin, d_iq_buff, count, iq_buff_size, iq_buff);
void multiArray2D_Wrapper(short* h_A, short* h_B, short* dev_CCos, short* dev_CSin, short* d_iq_buff, int rows, int cols, short* h_iq_buff)
{

	size_t size_array1D = cols * sizeof(short);
	size_t size_array2D = rows * cols * sizeof(short);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int blockSize = BLOCK_SIZES;
	int gridSize = (rows * cols + blockSize - 1) / blockSize;


	checkCuda(cudaMemcpy(dev_CCos, h_A, size_array2D, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_CSin, h_B, size_array2D, cudaMemcpyHostToDevice));//已经乘好的信号


	//kernelMultiArray2D << <gridSize, blockSize >> >(dev_ACos, dev_ASin, dev_B, dev_CCos, dev_CSin, rows, cols, gain);
	kernerSumColumnArray2D << <gridSize, blockSize >> > (dev_CCos, dev_CSin, rows, cols, d_iq_buff);
	//cudathreadsynchronize();

	checkCuda(cudaMemcpy(h_iq_buff, d_iq_buff, cols * 2 * sizeof(short), cudaMemcpyDeviceToHost));
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nthoi tian thuc hien copy bo nho: %f", milliseconds);

	printf("\nhet\n");
}
//折叠部分为存取纹理内存的核函数

__global__ void get_textureMem2D(int satnum, int* output)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;//ch_num=y,sample_num=x
	int idy = threadIdx.y;
	/*if (idy == satnum)
		printf("%d ", tex1D(t_CAcode, (float)idx + (float)idy * 1023));*/

	//while (idx < 2046 && idy == satnum)
	//{
	//	//float u = (float)idx / 1023;
	//	//float v = (float)idy / 10;
	//	output[idx] = tex1Dfetch(t_NH, (idx + idy * 2046));
	//	//output[idx] = cacode[idx+idy*1023];
	//	idx += blockDim.x * gridDim.x;
	//}
	while (idx < 20 && idy == 1)
	{
		/*int temp = tex1D(t_cosTable, float(idx));*/
		output[idx] = tex1D(t_NH, (float)idx);
		idx += blockDim.x * gridDim.x;
	}
}
/*
* 功能:存入经常使用的查询表和变量
* sin/cos table: texture memory
* pseudorandom code:texture memory
* navigation data:
* i_buff/q_buff:page-locked memory
* amplititude: texture memory
* input: sinTable,cosTable,CAcode
*/
//进行纹理内存的绑定，固定
void GPUMemoryInit(Table* Tb, int ch_num)
{
	int* output, * dev_output;
	int satnum = 0;//在和函数中使用，用于检查第satnum颗星星历或prn是否正常
	output = (int*)malloc(sizeof(int) * 2046);

	cudaChannelFormatDesc coschannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->cu_cosTable), &coschannelDesc, 512, 1);
	cudaMemcpyToArray(Tb->cu_cosTable, 0, 0, Tb->cosTable, sizeof(int) * 512, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_cosTable, Tb->cu_cosTable);//将余弦表绑定为纹理内存

	cudaChannelFormatDesc sinchannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->cu_sinTable), &sinchannelDesc, 512, 1);
	cudaMemcpyToArray(Tb->cu_sinTable, 0, 0, Tb->sinTable, sizeof(int) * 512, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_sinTable, Tb->cu_sinTable);//将正弦表绑定为纹理内存

	output = (int*)malloc(sizeof(int) * 2046);
	checkCuda(cudaMalloc((void**)&(dev_output), sizeof(int) * 2046));

	cudaChannelFormatDesc nhchannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->dev_NH), &nhchannelDesc, 20, 1);
	cudaMemcpyToArray(Tb->dev_NH, 0, 0, Tb->NH, sizeof(int) * 20, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_NH, Tb->dev_NH);//将NH码绑定为纹理内存

	checkCuda(cudaMalloc((void**)&Tb->dev_CAcode, sizeof(int) * 2046 * MAX_BDS_SAT));//把1023改成2046
	cudaMemcpy(Tb->dev_CAcode, Tb->CAcode, sizeof(int) * 2046 * MAX_BDS_SAT, cudaMemcpyHostToDevice);//把CA码全部绑定纹理内存
	cudaBindTexture(NULL, t_CAcode, Tb->dev_CAcode);//将CA码绑定为纹理内存

	checkCuda(cudaMalloc((void**)&Tb->dev_navdata, sizeof(char) * 1800 * MAX_BDS_SAT));//1800比特,一个帧包含5个子帧，一个子帧十个字，一个字30比特，一个比特占一个char,存储的时候把上一帧的第五子帧也保留下来，这样调用的时候方便一点
	cudaMemcpy(Tb->dev_navdata, Tb->navdata, sizeof(char) * 1800 * MAX_BDS_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_navbit, Tb->dev_navdata);//将导航电文绑定为纹理内存

	checkCuda(cudaMalloc((void**)&Tb->B1cCode.dev_datacode, sizeof(int) * 10230 * MAX_BDS_SAT));
	cudaMemcpy(Tb->B1cCode.dev_datacode, Tb->B1cCode.datacode, sizeof(int) * 10230 * MAX_BDS_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_B1c_data, Tb->B1cCode.dev_datacode);

	checkCuda(cudaMalloc((void**)&Tb->B1cCode.dev_pilotcode, sizeof(int) * 10230 * MAX_BDS_SAT));
	cudaMemcpy(Tb->B1cCode.dev_pilotcode, Tb->B1cCode.pilotcode, sizeof(int) * 10230 * MAX_BDS_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_B1c_pilot, Tb->B1cCode.dev_pilotcode);

	checkCuda(cudaMalloc((void**)&Tb->B1cCode.dev_secondcode, sizeof(int) * 1800 * MAX_BDS_SAT));
	cudaMemcpy(Tb->B1cCode.dev_secondcode, Tb->B1cCode.secondcode, sizeof(int) * 1800 * MAX_BDS_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_B1c_secondcode, Tb->B1cCode.dev_secondcode);

	checkCuda(cudaMalloc((void**)&Tb->dev_B1c_nav, sizeof(int) * 1800));
	cudaMemcpy(Tb->dev_B1c_nav, Tb->B1c_nav, sizeof(int) * 1800, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_B1c_nav, Tb->dev_B1c_nav);

}

void navdata_update(Table* Tb)
{
	checkCuda(cudaMalloc((void**)&Tb->dev_navdata, sizeof(char) * 1800 * MAX_BDS_SAT));
	cudaMemcpy(Tb->dev_navdata, Tb->navdata, sizeof(char) * 1800 * MAX_BDS_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_navbit, Tb->dev_navdata);//将导航电文绑定为纹理内存
}

__global__ void cudaBPSK(double* dev_parameters, cuFloatComplex* dev_i_buff, int* dev_sum, float* dev_noise,double *dev_db_para)//carrier_step
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;//ch_num=y,sample_num=x idx等于零的时候没有对码相位进行累加，实际上应该累加的
	int idy = threadIdx.y;//   idx指的是采样第几个点    idy是第几颗卫星
	int prn = dev_parameters[idy*p_n+2]-1;//在表中prn是从0开始的
	int sgn_sin[4] = { 1, 1, -1, -1 };
	int sgn_cos[4] = { 1,-1,-1,1 };
	double sqrt_Pb1i = 0.70710678118654752440084436210485;
	double sqrt_Pb1c = 0.70710678118654752440084436210485;
	double sqrt_a = 0.3015113446;//sqrt(1/11)
	double sqrt_b = 0.8118441409;//sqrt(29/44)
	double data_amp = 0.5;//sqrt(1/4)
	double pilot_amp = 0.8660;//sqrt(3/4)
	double pilot11_amp = 0.937436866561092 * pilot_amp;//sqrt(29/33)
	double pilot61_amp = 0.348155311911396 * pilot_amp;//sqrt(4/33)



	int CurrentCodePhase = (int)(idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0]) % 2046;//码相位=采样点数*码相位步进+初始码相位 可能会超出1023， 所以需要与1023取余
	double CarrierPhase = (idx * dev_db_para[idy * pd_n + 3] + dev_db_para[idy * pd_n + 2]); //相位的单位是周(2*pi) 载波相位=采样点数*载波相位步进+初始载波相位
	int cph = 0;
	int I_If = 0;
	if(CarrierPhase<0)
		 cph = (CarrierPhase - (int)CarrierPhase) * 512+512;//留下小数部分 
	else
		 cph = (CarrierPhase - (int)CarrierPhase) * 512;    //留下小数部分  零中频情况下出现了负值
	//int cph = (CarrierPhase >> 16) & 511;
	int temp11 = (int)((idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0]) / 2046.0 + (int)dev_parameters[idy * p_n + 4]%20);//得到ms数
	int ibit = temp11 / 20 + dev_parameters[idy*p_n+0]+ dev_parameters[idy * p_n + 3]*30;//前300bit为上一个帧的最后一个子帧，从301开始是当前子帧的第一个bit
	int NHnumber = (int)temp11 % 20;//NH码下标

	//子载波相位计算
	double IFcarr_phase = (idx * dev_db_para[idy * pd_n + 9]) + dev_db_para[idy * pd_n + 8];
	//IFcarr_phase = IFcarr_phase - (int)IFcarr_phase;

	//计算B1I 相位
	double B1I_phase = (idx * dev_db_para[idy * pd_n + 7] + dev_db_para[idy * pd_n + 6]);
	B1I_phase = B1I_phase - (int)B1I_phase;
	int B1I = B1I_phase * 512;


	//I_If = (IFcarr_phase - (int)IFcarr_phase) * 512;
	if (IFcarr_phase < 0)
		I_If = (IFcarr_phase - (int)IFcarr_phase) * 512 + 512;//留下小数部分 
	else
		I_If = (IFcarr_phase - (int)IFcarr_phase) * 512;    //留下小数部分  零中频情况下出现了负值

	IFcarr_phase = IFcarr_phase - (int)IFcarr_phase;
	
	double phase = CarrierPhase - IFcarr_phase;
	int iphase;
	if (phase < 0)
	{
		iphase = (phase - (int)phase) * 512 + 512;
	}
	else
	{
		iphase = (phase - (int)phase) * 512;
	}
	phase = phase - (int)phase;
	//b1c信号对应码相位等参数计算
	int B1c_CodePhase = (idx * dev_db_para[idy * pd_n + 5] + dev_db_para[idy * pd_n + 4]);
	B1c_CodePhase = B1c_CodePhase % 10230;

	int B1c_ibit = (int)(idx * dev_db_para[idy * pd_n + 5] + dev_db_para[idy * pd_n + 4]) / 10230 + dev_parameters[idy * pd_n + 5];
	B1c_ibit = B1c_ibit % 1800;

	double fa_phase = (idx * dev_db_para[idy * pd_n + 11] + dev_db_para[idy * pd_n + 10]);
	fa_phase = fa_phase - (int)fa_phase;

	double fb_phase = (idx * dev_db_para[idy * pd_n + 13] + dev_db_para[idy * pd_n + 12]);
	fb_phase = fb_phase - (int)fb_phase;

	int secondcode = dev_parameters[idy * pd_n + 6] + (int)(idx * dev_db_para[idy * pd_n + 5] + dev_db_para[idy * pd_n + 4]) / 10230;
	secondcode = secondcode % 1800;

	__shared__ cuFloatComplex memoryi[MAX_CHAN][threadPerBlock];//本机GPU每线程块内的共享内存为48KB
	cuFloatComplex result1, result2, result3, result4;
	
	double temp1, temp2, temp3, temp4, temp5;
	double temp1_i;
	int num, num1, num2;
	if (idx < dev_sum[0])//
	{
		cuFloatComplex result1,result2;
		num = (int)(IFcarr_phase * 4) % 4;
		num1 = (int)(fa_phase * 4) % 4;
		num2 = (int)(fb_phase * 4) % 4;
		double result;


		//temp1 = tex1Dfetch(t_navbit, (ibit + prn * 1800)) * tex1D(t_NH, (float)NHnumber) * -1 * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046));
		//result1 = cuCmulf(make_cuFloatComplex(sqrt_Pb1i * temp1, 0), make_cuFloatComplex(sgn_cos[num], sgn_sin[num] * -1));
		//temp2 = tex1Dfetch(t_B1c_nav, B1c_ibit) * tex1Dfetch(t_B1c_data, B1c_CodePhase + prn * 10230) * sgn_sin[num1];
		//temp2 = 0.5 * sqrt_Pb1c * temp2;
		//temp3 = tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num2];
		//temp3 = sqrt_Pb1c * sqrt_a * temp3;
		//temp4 = tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num1];
		//temp4 = sqrt_Pb1c * sqrt_b * temp4;
		//result2 = cuCaddf(result1, make_cuFloatComplex((temp2 + temp3), temp4));
		//memoryi[idy][threadIdx.x] = cuCmulf(result2, make_cuFloatComplex(tex1D(t_cosTable, (float)cph) / 250.0, tex1D(t_sinTable, (float)cph) / 250.0));

		//temp1 = tex1Dfetch(t_navbit, (ibit + prn * 1800)) * tex1D(t_NH, (float)NHnumber) * -1 * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046));
		//temp2 = sgn_cos[num] * tex1D(t_cosTable, (float)cph) / 250.0 + sgn_sin[num] * tex1D(t_sinTable, (float)cph) / 250.0;
		//temp3 = tex1Dfetch(t_B1c_nav, B1c_ibit) * tex1Dfetch(t_B1c_data, B1c_CodePhase + prn * 10230) * sgn_sin[num1] * tex1D(t_cosTable, (float)cph) / 250.0;
		//temp4 = tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num2] * tex1D(t_cosTable, (float)cph) / 250.0;
		//temp5 = tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num1] * tex1D(t_sinTable, (float)cph) / 250.0;
		//result = sqrt_Pb1i * temp1 * temp2 + 0.5 * sqrt_Pb1c * temp3 + sqrt_Pb1c * sqrt_a * temp4 - sqrt_Pb1c * sqrt_b * temp5;
		//memoryi[idy][threadIdx.x] = cuCmulf(make_cuFloatComplex(result, 0), make_cuFloatComplex(tex1D(t_cosTable, (float)cph) / 250.0, tex1D(t_sinTable, (float)cph) / 250.0));
		// 
		// 
		// 
		
		//temp1 = data_amp * tex1Dfetch(t_B1c_nav, B1c_ibit) * tex1Dfetch(t_B1c_data, B1c_CodePhase + prn * 10230) * sgn_sin[num1];
		//temp2 = pilot61_amp * tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num2];
		//temp3 = pilot11_amp * tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num1];
		//temp4 = tex1Dfetch(t_navbit, (ibit + prn * 1800)) * tex1D(t_NH, (float)NHnumber) * -1 * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046));
		//result1 = cuCmulf(make_cuFloatComplex(dev_parameters[p_n * idy + 1], 0), make_cuFloatComplex((temp1 + temp2), temp3));
		//result2 = cuCmulf(make_cuFloatComplex(temp4 * dev_parameters[p_n * idy + 1], 0), make_cuFloatComplex(tex1D(t_cosTable, (float)I_If) / 250.0, (tex1D(t_sinTable, (float)I_If) / 250.0) * -1));
		//memoryi[idy][threadIdx.x] = cuCmulf(cuCaddf(result1, result2), make_cuFloatComplex(tex1D(t_cosTable, (float)cph) / 250.0, tex1D(t_sinTable, (float)cph) / 250.0));
	
	
		temp1 = data_amp * tex1Dfetch(t_B1c_nav, B1c_ibit) * tex1Dfetch(t_B1c_data, B1c_CodePhase + prn * 10230) * sgn_sin[num1];
		temp2 = pilot61_amp * tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num2];
		temp3 = pilot11_amp * tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230) * tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800) * sgn_sin[num1];
		temp4 = tex1Dfetch(t_navbit, (ibit + prn * 1800)) * tex1D(t_NH, (float)NHnumber) * -1 * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046));
		result1 = cuCmulf(make_cuFloatComplex(dev_parameters[p_n * idy + 1], 0), make_cuFloatComplex((temp1 + temp2), temp3));
		result2 = cuCmulf(make_cuFloatComplex(temp4 * dev_parameters[p_n * idy + 1], 0), make_cuFloatComplex(tex1D(t_cosTable, (float)B1I) / 250.0, (tex1D(t_sinTable, (float)B1I) / 250.0)));
		memoryi[idy][threadIdx.x] = cuCmulf(result1, make_cuFloatComplex(tex1D(t_cosTable, (float)cph) / 250.0, tex1D(t_sinTable, (float)cph) / 250.0));
		//memoryi[idy][threadIdx.x] = make_cuFloatComplex(0, 0);
		memoryi[idy][threadIdx.x] = cuCaddf(memoryi[idy][threadIdx.x], result2);
	}
	__syncthreads();


	//if ((idx == 1212) && idy == 0)
	//{
	//	double temp11 = tex1D(t_cosTable, (float)B1I) / 250.0;
	//	printf("\nidx==%d!!!!!!!!!!!!!!!!!\n", idx);
	//	printf("\nresult.x=%f\n", memoryi[idy][threadIdx.x].x);
	//	printf("t_navbit:%d\n", tex1Dfetch(t_navbit, (ibit + prn * 1800)));
	//	printf("ibit:%d\n", ibit);
	//	printf("t_NH:%d\n", tex1D(t_NH, (float)NHnumber));
	//	printf("t_CAcode:%d\n", tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046)));
	//	printf("CurrentCodePhase:%d\n\n", CurrentCodePhase);
	//	printf("初始码相位：%f\n", dev_db_para[idy * pd_n + 0]);
	//	printf("码相位步进：%f\n\n", dev_db_para[idy * pd_n + 1]);
	//	printf("dev_parameters[p_n * idy + 1]:%f\n", dev_parameters[p_n * idy + 1]);
	//	printf("B1I:%d\n", B1I);
	//	printf("t_cosTable:%f\n", temp11);
	//	printf("\nB1I初始载波相位：%10.10f\n", dev_db_para[idy * pd_n + 6]);
	//	printf("B1I载波相位步进：%10.10f\n\n", dev_db_para[idy * pd_n + 7]);
	//}

	//if ((idx == 1 ) && idy == 0)
	//{
	//	printf("\nidx==%d        result.x=%f\n\n",idx, memoryi[idy][threadIdx.x].x);
	//	//printf("\nt_navbit:%d\n", tex1Dfetch(t_navbit, (ibit + prn * 1800)));
	//	//printf("t_NH:%d\n", tex1D(t_NH, (float)NHnumber));
	//	//printf("t_CAcode:%d\n", tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046)));
	//	//printf("dev_parameters[p_n * idy + 1]:%f\n", dev_parameters[p_n * idy + 1]);
	//	//printf("t_cosTable:%f\n", tex1D(t_cosTable, (float)cph) / 250);
	//	//printf("CarrierPhase:%f\n", CarrierPhase);
	//	//printf("初始相位:%f\n", dev_db_para[idy * pd_n + 2]);
	//	//printf("相位步进:%f\n", dev_db_para[idy * pd_n + 3]);
	//	//printf("cph:%d\n\n", cph);
	//}



	//if (idx == 1e6 && idy == 0)
	//{
	//	printf("t_B1c_nav:%d\n", tex1Dfetch(t_B1c_nav, B1c_ibit));
	//	printf("t_B1c_data:%d\n", tex1Dfetch(t_B1c_data, B1c_CodePhase + prn * 10230));
	//	printf("t_B1c_pilot:%d\n", tex1Dfetch(t_B1c_pilot, B1c_CodePhase + prn * 10230));
	//	printf("t_B1c_secondcode:%d\n", tex1Dfetch(t_B1c_secondcode, secondcode + prn * 1800));
	//	printf("B1c_ibit:%d\n", B1c_ibit);//
	//	printf("B1c_CodePhase:%d\n", B1c_CodePhase);//
	//	printf("secondcode:%d\n", secondcode);//
	//	printf("prn:%d", prn);
	//	printf("dev_db_para[idy * pd_n + 5]:%f\n", dev_db_para[idy * pd_n + 5]);
	//	printf("dev_db_para[idy * pd_n + 4]:%f\n", dev_db_para[idy * pd_n + 4]);
	//}
	//if (( idx==1||idx == dev_sum[0]) && idy == 2)
	//{
	//	printf("\nidx==%d!!!!!!!!!!!!\n", idx);
	//	printf("iB1I:%d\n", B1I);
	//	printf("B1I_phase:%10.10f\n", B1I_phase);
	//	printf("B1I初始载波相位：%10.10f\n", dev_db_para[idy * pd_n + 6]);
	//	printf("B1I载波相位步进：%10.10f\n", dev_db_para[idy * pd_n + 7]);
	//	printf("iB1C:%d\n", cph);
	//	printf("B1C_phase:%10.10f\n", CarrierPhase - (int)CarrierPhase);
	//	printf("B1C初始载波相位：%10.10f\n", dev_db_para[idy * pd_n + 2]);
	//	printf("B1C载波相位步进：%10.10f\n\n\n", dev_db_para[idy * pd_n + 3]);
	//	//printf("B1c_ibit:%d\n", B1c_ibit);
	//	//printf("B1c_CodePhase:%d\n", B1c_CodePhase);
	//	//printf("B1C初始码相位：%f\n", dev_db_para[idy * pd_n + 4]);
	//	//printf("B1C码相位步进：%f\n\n", dev_db_para[idy * pd_n + 5]);
	//	//printf("secondcode:%d\n\n", secondcode);
	//	//printf("fa_phase:%f\n", fa_phase);
	//	//printf("fb_phase:%f=idx*%f+%f\n\n", fb_phase, dev_db_para[idy * pd_n + 13], dev_db_para[idy * pd_n + 12]);

	//}

	//if ((idx == 1||idx==16.368e5) && idy == 2)
	//{
	//	cuFloatComplex temp = cuCmulf(make_cuFloatComplex(tex1D(t_cosTable, (float)I_If) / 250.0, (tex1D(t_sinTable, (float)I_If) / 250.0) * -1), make_cuFloatComplex(tex1D(t_cosTable, (float)cph) / 250.0, tex1D(t_sinTable, (float)cph) / 250.0));
	//	printf("idx==%d!!!!!!!!!!!!\n", idx);
	//	printf("\nt_navbit:%d\n", tex1Dfetch(t_navbit, (ibit + prn * 1800)));
	//	printf("prn:%d", prn);
	//	printf("t_NH:%d\n", tex1D(t_NH, (float)NHnumber));
	//	printf("t_CAcode:%d\n", tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 2046)));
	//	printf("dev_parameters[p_n * idy + 1]:%f\n", dev_parameters[p_n * idy + 1]);
	//	//printf("t_cosTable:%f\n", temp.x);
	//	printf("iphase:%d\n", iphase);
	//	printf("phase:%f\n", phase);
	//	printf("初始相位:%f==%f-%f\n", dev_db_para[idy * pd_n + 2] - dev_db_para[idy * pd_n + 8], dev_db_para[idy * pd_n + 2], dev_db_para[idy * pd_n + 8]);
	//	printf("iB1I=%d\n", B1I);
	//	printf("carr_step=%f=%f-%f\n\n\n", dev_db_para[idy * pd_n + 3] - dev_db_para[idy * pd_n + 9], dev_db_para[idy * pd_n + 3], dev_db_para[idy * pd_n + 9]);
	//}



	//if (idx == 1000 && idy == 0)
	//{
	//	printf("载波相位：%f\n", CarrierPhase);
	//	printf("载波相位步进：%f\n", dev_db_para[idy * pd_n + 3]);
	//	printf("B1I码相位：%d\n", CurrentCodePhase);
	//	printf("B1I码步进：%f\n", dev_db_para[idy * pd_n + 1]);
	//	printf("B1C码相位：%d\n", B1c_CodePhase);
	//	printf("B1C码相位步进：%f\n", dev_db_para[idy * pd_n + 5]);
	//	printf("B1I子载波相位：%f=%f*%d+%f\n", IFcarr_phase, dev_db_para[idy * pd_n + 9], idx, dev_db_para[idy * pd_n + 8]);
	//	printf("B1I子载波步进：%f\n", dev_db_para[idy * pd_n + 9]);
	//	printf("fa载波相位：%f\n", fa_phase);
	//	printf("fa载波相位步进：%f\n", dev_db_para[idy * pd_n + 11]);
	//	printf("fb载波相位：%f\n", fb_phase);
	//	printf("fb载波相位步进：%f\n",dev_db_para[idy * pd_n + 13]);
	//}

	//这里使用归约运算
	int i = dev_sum[1] / 2;
	while (i !=0)//只进行一次归约
	{
		if (idy < i)
			memoryi[idy][threadIdx.x] = cuCaddf(memoryi[idy][threadIdx.x],memoryi[idy + i][threadIdx.x]);//将各个卫星信号进行累加，最后都加到第一个卫星存的地方了
		__syncthreads();
		i /= 2;
	}
	//for (int i = 1; i <= dev_sum[1]; i++)
	//{
	//	memoryi[idy][threadIdx.x] = cuCaddf(memoryi[idy][threadIdx.x], memoryi[idy + i][threadIdx.x]);
	//}
	// 
	// 
	//if(idy==1)
	//int num1 = idx * 2;
	if (idy == 0)
	{
		//dev_i_buff[idx] = cuCaddf(memoryi[0][threadIdx.x], make_cuFloatComplex(dev_noise[idx], dev_noise[idx+ dev_sum[0]]));  //Short((memoryi[0][threadIdx.x] + 64) >> 7)   dev_noise[idx]
		dev_i_buff[idx] = memoryi[0][threadIdx.x];
	}
}
void produce_samples_withCuda(Table* Tb, channel_t* channel, int fs, double* parameters, double* dev_parameters, int* sum, int* dev_sum, float* dev_i_buff, int satnum, float* dev_noise, double* db_para, double* dev_db_para)
{
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	int sat_all = satnum;
	float * dev_amplitude, * amplitude;
	int samples = fs / Rev_fre;//
	cuFloatComplex* dev_buff;
	for (int j = 0; j < sat_all; j++, channel++)
	{
		if (channel->prn != 0)
		{
			parameters[j * p_n + 0] = channel->ibit;
			parameters[j * p_n + 1] = channel->amp;
			parameters[j * p_n + 2] = channel->prn;
			parameters[j * p_n + 3] = channel->iword;
			parameters[j * p_n + 4] = channel->icode;
			parameters[j * p_n + 5] = channel->b1c.B1c_ibit;
			parameters[j * p_n + 6] = channel->b1c.B1c_icode;
			db_para[j * pd_n + 0] = channel->code_phase;
			db_para[j * pd_n + 1] = channel->code_phasestep;
			db_para[j * pd_n + 2] = channel->carr_phase;
			db_para[j * pd_n + 3] = channel->carr_phasestep;
			db_para[j * pd_n + 4] = channel->b1c.B1c_CodePhase;
			db_para[j * pd_n + 5] = channel->b1c.B1c_CodeStep;
			db_para[j * pd_n + 6] = channel->B1I_carr;
			db_para[j * pd_n + 7] = channel->B1I_carrstep;
			db_para[j * pd_n + 8] = channel->IFcarr_phase;
			db_para[j * pd_n + 9] = channel->IFcarr_step;
			db_para[j * pd_n + 10] = channel->b1c.fa_phase;
			db_para[j * pd_n + 11] = channel->b1c.fa_step;
			db_para[j * pd_n + 12] = channel->b1c.fb_phase;
			db_para[j * pd_n + 13] = channel->b1c.fb_step;
		}
	}
	//
	checkCuda(cudaMalloc((void**)&dev_buff, samples * sizeof(cuFloatComplex) * 1));//checkCuda(cudaMalloc((void**)&dev_buff, samples * sizeof(float) * 2));
	checkCuda(cudaMemcpy(dev_parameters, parameters, sizeof(double) * sat_all * p_n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_db_para, db_para, sizeof(double) * sat_all * pd_n, cudaMemcpyHostToDevice));

	sum[0] = samples; sum[1] = MAX_CHAN;
  	checkCuda(cudaMemcpy(dev_sum, &sum[0], 2 * sizeof(int), cudaMemcpyHostToDevice));//
	float blockPerGridx = fs / Rev_fre / threadPerBlock;
	(blockPerGridx - (int)blockPerGridx == 0) ? blockPerGridx : blockPerGridx = (int)blockPerGridx + 1;
	dim3 block(blockPerGridx, 1);
	dim3 thread(threadPerBlock, satnum);
	cudaBPSK << <block, thread >> > (dev_parameters, dev_buff, dev_sum,dev_noise,dev_db_para);

	checkCuda(cudaMemcpy(Tb->buff, dev_buff, sizeof(cuFloatComplex) * samples * 1, cudaMemcpyDeviceToHost));




		//printf("\n%f    %f", Tb->buff[1099373].x, Tb->buff[1099373].y);



	cudaFree(dev_buff);
}

__global__ void kernalQuantify(float* iq_buff, unsigned char* out_buff, int iq_buff_size, float th)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float src = iq_buff[idx];
	int bcount = idx / 4;
	unsigned char ibit = ibit & 0x00;
	__shared__ unsigned char memory[16];
	if (src > th)
	{
		ibit = 0x01;//3
	}
	else if (src < th && src>0)
	{
		ibit = 0x00;//1
	}
	else if (src < -th)
	{
		ibit = 0x03;//-3
	}
	else if (src > -th && src < 0)
	{
		ibit = 0x02;//-1
	}
	memory[threadIdx.x] = ibit;
	__syncthreads();
	if (threadIdx.x % 4 == 0)
		out_buff[bcount] = memory[threadIdx.x] << 6 | memory[1 + threadIdx.x] << 4 | memory[2 + threadIdx.x] << 2 | memory[3 + threadIdx.x];
}
void quantify_with_cuda(float* dev_iq_buff, float* iq_buff, unsigned char* out_buff, unsigned char* dev_out_buff, int iq_buff_size, float* th)
{
	checkCuda(cudaMemcpy(dev_iq_buff, iq_buff, sizeof(float) * iq_buff_size * 2, cudaMemcpyHostToDevice));
	int blockxPergrid = iq_buff_size * 2 / 16 + 1;
	dim3 block(blockxPergrid, 1);
	dim3 thread(16, 1);
	kernalQuantify << <block, thread >> > (dev_iq_buff, dev_out_buff, iq_buff_size, *th);

	checkCuda(cudaMemcpy(out_buff, dev_out_buff, iq_buff_size / 2, cudaMemcpyDeviceToHost));
}

void GPUMemroy_delete(Table* Tb)
{
	cudaUnbindTexture(t_sinTable);
	cudaUnbindTexture(t_cosTable);
	cudaUnbindTexture(t_CAcode);
	cudaFree(Tb->dev_CAcode);
	cudaFreeArray(Tb->cu_cosTable);
	cudaFreeArray(Tb->cu_sinTable);
	cudaFreeArray(Tb->dev_NH);
	//free(Tb->i_buff);
}





//__global__ void kernel(channel_t *chan, int *gain, double delt, int count, int iq_buff_size, short *iq_buff) {
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if (idx < iq_buff_size) {
//		int ip, qp, i_acc, q_acc;
//		int iTable;
//		i_acc = 0;
//		q_acc = 0;
//		for (int i = 0; i < count; i++) {
//			if (chan[i].prn > 0) {
//
//				iTable = (chan[i].carr_phase >> 16) & 511;
//
//				ip = chan[i].dataBit * chan[i].codeCA * cosTable512[iTable] * gain[i];
//				qp = chan[i].dataBit * chan[i].codeCA * sinTable512[iTable] * gain[i];
//
//				i_acc += ip;
//				q_acc += qp;
//
//				chan[i].code_phase += chan[i].f_code * delt;
//
//				if (chan[i].code_phase >= CA_SEQ_LEN) {
//
//					chan[i].code_phase -= CA_SEQ_LEN;
//					chan[i].icode++;
//
//					if (chan[i].icode >= 20) { // 20 C/A codes = 1 navigation data bit
//						chan[i].icode = 0;
//						chan[i].ibit++;
//
//						if (chan[i].ibit >= 30) { // 30 navigation data bits = 1 word
//							chan[i].ibit = 0;
//							chan[i].iword++;
//							/*
//							if (chan[i].iword>=N_DWRD)
//							printf("\nWARNING: Subframe word buffer overflow.\n");
//							*/
//						}
//
//						// Set new navigation data bit
//						chan[i].dataBit = (int)((chan[i].dwrd[chan[i].iword] >> (29 - chan[i].ibit)) & 0x1UL) * 2 - 1;
//					}
//				}
//
//				// Set currnt code chip
//				chan[i].codeCA = chan[i].ca[(int)chan[i].code_phase] <<1- 1;
//
//				// Update carrier phase
//				chan[i].carr_phase += chan[i].carr_phasestep;
//			}
//		}
//
//
//		// Scaled by 2^7
//		i_acc = (i_acc + 64) >> 7;
//		q_acc = (q_acc + 64) >> 7;
//
//		// Store I/Q samples into buffer
//		iq_buff[idx * 2] = (short)i_acc;
//		iq_buff[idx * 2 + 1] = (short)q_acc;
//	}
//}
//
//extern "C" void handleData_in_kernel(channel_t *chan, int *gain, double delt, int count, int iq_buff_size, short *iq_buff) {
//	
//	size_t sizeChannel = count * sizeof(channel_t);
//	size_t sizeIq_buff = iq_buff_size * 2 * sizeof(short);
//	size_t sizeGain = count * sizeof(int);
//	
//	int *dev_gain;
//	short *dev_iq_buff;
//	channel_t *dev_chan;
//
//	int blockSize = 1024;
//	int gridSize = (iq_buff_size + blockSize - 1) / blockSize;
//
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start);
//
//	checkCuda(cudaMalloc((void**)&dev_chan, sizeChannel));
//	checkCuda(cudaMalloc((void**)&dev_gain, sizeGain));
//	checkCuda(cudaMalloc((void**)&dev_iq_buff, sizeIq_buff));
//
//	checkCuda(cudaMemcpy(dev_gain, gain, sizeGain, cudaMemcpyHostToDevice));
//	checkCuda(cudaMemcpy(dev_chan, chan, sizeChannel, cudaMemcpyHostToDevice));
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\nthoi tian thuc hien copy bo nho: %f", milliseconds);
//
//	cudaEventRecord(start);
//	kernel << < gridSize, blockSize>> > (dev_chan, dev_gain, delt, count, iq_buff_size, dev_iq_buff);
//	cudaThreadSynchronize();
//
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds2 = 0;
//	cudaEventElapsedTime(&milliseconds2, start, stop);
//	printf("\nthoi tian thuc hien trong gpu: %f", milliseconds2);
//	checkCuda(cudaMemcpy(iq_buff, dev_iq_buff, sizeIq_buff, cudaMemcpyDeviceToHost));
//
//	checkCuda(cudaFree(dev_chan));
//	checkCuda(cudaFree(dev_gain));
//	checkCuda(cudaFree(dev_iq_buff));
//}