#include "gpssim.h"
#include <stdio.h>
#include <complex>
#include "kernal.h"
#include "B1cCodeNav.h"
#include <string.h>


int main()
{
	const char* rfile = "../BRDC00GOP_R_20220070000_01D_MN.rnx";//星历文件
	const char* tfile = "../tra.csv";//用户轨迹
	FILE* feph1 = NULL;
	feph1 = fopen("D:\\B1Iresult_no_noise.txt", "w");
	//feph1 = fopen("D:\\SCBOCresult.bin", "wb");
	typedef std::complex<float> complexf;
	int generated_samples = 0;//一共产生了多少采样点
	int samp_freq = 32736000;//16.368e6*2;//采样率16.368  49104000
	int simu_time = 10; //调用一次bds_sim函数，产生多久的信号（单位0.1s）
	int time_all = 10;//总仿真时间（单位0.1s）
	int buff_size = (samp_freq * simu_time / 10);//存放的采样点的个数
	cuFloatComplex*buff = new cuFloatComplex[buff_size];//用于存放输出的中频信号
	//int* qua_buff = (int*)malloc(sizeof(int) * buff_size / 5);
	int ibit = 0;
	transfer_parameter tp;
	bdstime_t g0;
	tp.g0.week = -1;
	g0.sec = 10.0;
	tp.xyz = (double(*)[3]) malloc(sizeof(double) * time_all * 3);//初始化存放用户位置的内存
	tp.navbit = (char*)malloc(sizeof(char) * MAX_BDS_SAT * 1800);//初始化存放星历的内存(B1I)
	tp.b1c_nav = (char*)malloc(sizeof(char) * MAX_BDS_SAT * 1800);//b1c信号电文
	tp.neph = read_BDS_RinexNav_All(tp.eph, &tp.ionoutc, rfile);//读取b1i广播星历，存放到“eph”中，电离层信息存放到“ionoutc”中，返回值为广播星历持续的时间
	readUserMotion(tp.xyz, tfile, time_all);//读取仿真时间内的用户轨迹，存放到xyz的二维数组中
	Table BDStable;
	BDStable.buff = new cuFloatComplex[buff_size];
	int* CAcode;
	CAcode = (int*)malloc(sizeof(int) * MAX_BDS_SAT * 2046 );//产生b1i信号的扩频码
	for (int i = 0; i < MAX_BDS_SAT; i++)
	{
		BDB1Icodegen((CAcode + i * 2046 ), i + 1);//产生所有卫星的C/A码
	}
	BDStable.CAcode = CAcode;//存放到表中

	BDStable.B1cCode.datacode = (int*)malloc(sizeof(int) * MAX_BDS_SAT * 10230);
	BDStable.B1cCode.pilotcode= (int*)malloc(sizeof(int) * MAX_BDS_SAT * 10230);
	BDStable.B1cCode.secondcode= (int*)malloc(sizeof(int) * MAX_BDS_SAT * 1800);
	for (int i = 0; i < MAX_BDS_SAT;i++)
	{
		BDSB1Ccodegen((BDStable.B1cCode.datacode + i * 10230), (BDStable.B1cCode.pilotcode + i * 10230), (BDStable.B1cCode.secondcode + i * 1800), i);
	}
	
	BDStable.i_buff = (float*)malloc(2 * sizeof(float) * buff_size);
	float *dev_i_buff=NULL;

	//FILE* fpw;
	//fpw = fopen("..\\motion.txt", "w+");
	//fprintf(fpw, "北斗时    x   y   z\n");

	for (int i = 0; i < (int)(time_all / simu_time); i++)
	{
		//memset(qua_buff, 0, sizeof(int) * buff_size / 5);
		generated_samples = bds_sim(&BDStable, &tp, buff, buff_size, samp_freq, 7161000*1, simu_time, dev_i_buff);//输入相应参数，将信号存放到buff数组中返回
		//if (i == 0)
		//{
		//	printf("作为对比\n");
		//	for (int i = 0; i < 10; i++)
		//	{
		//		printf("\n%f    %f", buff[i].x, buff[i].y);
		//	}
		//}
		
		
		
		if(i==0)
		{
			//fwrite(buff, 8, (int)(buff_size * 0.9), feph1);
			for (int i = 0; i < (int)(buff_size*0.9); i++)
			{
			    fprintf(feph1, "%10f   %10f\n", buff[i].x, buff[i].y);
			}
		}
		else
		{
			//fwrite(buff, 8, (int)(buff_size), feph1);
			for (int i = 0; i < (int)(buff_size); i++)
			{
				fprintf(feph1, "%10f   %10f\n", buff[i].x, buff[i].y);
			}
		}



		//if (i == 1)
		//{
		//	for (int j =5e5; j < 5e5+100; j++)
		//	{
		//		printf("%.5f %.5f\n", buff[j].real(), buff[j].imag());
		//	}
		//	printf("\n\n");
		//}
	}
	fclose(feph1);
	//fclose(fpw);
	delete[]buff;
	//free(qua_buff);
	free(tp.navbit);
	free(tp.xyz);
	int aa=getchar();
}