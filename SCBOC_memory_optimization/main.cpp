#include "gpssim.h"
#include <stdio.h>
#include <complex>
#include "kernal.h"
#include "B1cCodeNav.h"
#include <string.h>


int main()
{
	const char* rfile = "../BRDC00GOP_R_20220070000_01D_MN.rnx";//�����ļ�
	const char* tfile = "../tra.csv";//�û��켣
	FILE* feph1 = NULL;
	feph1 = fopen("D:\\B1Iresult_no_noise.txt", "w");
	//feph1 = fopen("D:\\SCBOCresult.bin", "wb");
	typedef std::complex<float> complexf;
	int generated_samples = 0;//һ�������˶��ٲ�����
	int samp_freq = 32736000;//16.368e6*2;//������16.368  49104000
	int simu_time = 10; //����һ��bds_sim������������õ��źţ���λ0.1s��
	int time_all = 10;//�ܷ���ʱ�䣨��λ0.1s��
	int buff_size = (samp_freq * simu_time / 10);//��ŵĲ�����ĸ���
	cuFloatComplex*buff = new cuFloatComplex[buff_size];//���ڴ���������Ƶ�ź�
	//int* qua_buff = (int*)malloc(sizeof(int) * buff_size / 5);
	int ibit = 0;
	transfer_parameter tp;
	bdstime_t g0;
	tp.g0.week = -1;
	g0.sec = 10.0;
	tp.xyz = (double(*)[3]) malloc(sizeof(double) * time_all * 3);//��ʼ������û�λ�õ��ڴ�
	tp.navbit = (char*)malloc(sizeof(char) * MAX_BDS_SAT * 1800);//��ʼ������������ڴ�(B1I)
	tp.b1c_nav = (char*)malloc(sizeof(char) * MAX_BDS_SAT * 1800);//b1c�źŵ���
	tp.neph = read_BDS_RinexNav_All(tp.eph, &tp.ionoutc, rfile);//��ȡb1i�㲥��������ŵ���eph���У��������Ϣ��ŵ���ionoutc���У�����ֵΪ�㲥����������ʱ��
	readUserMotion(tp.xyz, tfile, time_all);//��ȡ����ʱ���ڵ��û��켣����ŵ�xyz�Ķ�ά������
	Table BDStable;
	BDStable.buff = new cuFloatComplex[buff_size];
	int* CAcode;
	CAcode = (int*)malloc(sizeof(int) * MAX_BDS_SAT * 2046 );//����b1i�źŵ���Ƶ��
	for (int i = 0; i < MAX_BDS_SAT; i++)
	{
		BDB1Icodegen((CAcode + i * 2046 ), i + 1);//�����������ǵ�C/A��
	}
	BDStable.CAcode = CAcode;//��ŵ�����

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
	//fprintf(fpw, "����ʱ    x   y   z\n");

	for (int i = 0; i < (int)(time_all / simu_time); i++)
	{
		//memset(qua_buff, 0, sizeof(int) * buff_size / 5);
		generated_samples = bds_sim(&BDStable, &tp, buff, buff_size, samp_freq, 7161000*1, simu_time, dev_i_buff);//������Ӧ���������źŴ�ŵ�buff�����з���
		//if (i == 0)
		//{
		//	printf("��Ϊ�Ա�\n");
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