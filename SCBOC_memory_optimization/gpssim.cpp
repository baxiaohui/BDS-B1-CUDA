#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


#include "kernal.h"

#ifdef _WIN32
#include "getopt.h"
#else
#include <unistd.h>
#endif
#include "gpssim.h"
//TPoint *point;
int countPoint = 0;
//正弦表
int sinTable512[] = {
	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
};


//余弦表
int cosTable512[] = {
	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
};
//20bit NH码 为方便异或计算，把0换成-1
int NH[20] = { -1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1};

// Receiver antenna attenuation in dB for boresight angle = 0:5:180 [deg]//boresight 视轴是天线增益最大的地方，所以零度的时候为0，此时增益最高
double ant_pat_db[37] = {
	0.00,  0.00,  0.22,  0.44,  0.67,  1.11,  1.56,  2.00,  2.44,  2.89,  3.56,  4.22,
	4.89,  5.56,  6.22,  6.89,  7.56,  8.22,  8.89,  9.78, 10.67, 11.56, 12.44, 13.33,
	14.44, 15.56, 16.67, 17.78, 18.89, 20.00, 21.33, 22.67, 24.00, 25.56, 27.33, 29.33,
	31.56
};



/*! \brief Subtract two vectors of double
*  \param[out] y Result of subtraction
*  \param[in] x1 Minuend of subtracion
*  \param[in] x2 Subtrahend of subtracion
*/
void subVect(double *y, const double *x1, const double *x2)//两个向量做减法
{
	y[0] = x1[0] - x2[0];
	y[1] = x1[1] - x2[1];
	y[2] = x1[2] - x2[2];

	return;
}
/// <mgrn-生成服从高斯分布的随机数>
/// 
/// </summary>
/// <param name="u"></param>
/// <param name="g"></param>
/// <param name="r"></param>
/// <returns></随机噪声>
double mgrn1(double u, double g, double* r)//产生满足正态分布的随机数
{
	int i, m;
	double s, w, v, t;
	s = 65536.0; w = 2053.0; v = 13849.0;
	t = 0.0;
	for (i = 1; i <= 12; i++)
	{
		*r = (*r) * w + v;
		m = (int)(*r / s);
		*r = *r - m * s; 
		t = t + (*r) / s;
	}
	t = u + g * (t - 6.0);
	return(t);
}

//double mpow(double base, double index)
//{
//	double temp = base;
//	for (int i = 0; i < index-1; i++)
//	{
//		base = base * temp;
//	}
//	return base;
//}
/*! \brief Compute Norm of Vector
*  \param[in] x Input vector
*  \returns Length (Norm) of the input vector
*/
double normVect(const double *x)//计算向量二范数
{
	return(sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
}

/*! \brief Compute dot-product of two vectors
*  \param[in] x1 First multiplicand
*  \param[in] x2 Second multiplicand
*  \returns Dot-product of both multiplicands
*/
double dotProd(const double *x1, const double *x2)//计算两个向量的点积
{
	return(x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2]);
}

/* !\brief generate the C/A code sequence for a given Satellite Vehicle PRN
*  \param[in] prn PRN nuber of the Satellite Vehicle
*  \param[out] ca Caller-allocated integer array of 1023 bytes
*/
//产生北斗的C/A码 直接添加NH调制,直接产生20*2046个码片
void BDB1Icodegen(int* out, int svnum)//产生C/A码
{
	int gs2[] = { 1, 3, 1, 4, 1, 5, 1, 6, 1, 8, 1, 9, 1, 10, 1, 11, 2, 7, 3, 4, 3, 5, 3, 6, 3, 8, 3, 9, 3, 10, 3, 11, 4, 5, 4, 6, 4, 8, 4, 9, 4, 10, 4, 11, 5, 6, 5, 8, 5, 9, 5, 10, 5, 11, 6, 8, 6, 9, 6, 10, 6, 11, 8, 9, 8, 10, 8, 11, 9, 10, 9, 11, 10, 11 };
	int gs2_continue[26][3] = { {1, 2, 7},
		{1, 3, 4},
		{1, 3, 6},
		{1, 3, 8},
		{1, 3, 10},
		{1, 3, 11},
		{1, 4, 5},
		{1, 4, 9},
		{1, 5, 6},
		{1, 5, 8},
		{1, 5, 10},
		{1, 5, 11},
		{1, 6, 9},
		{1, 8, 9},
		{1, 9, 10},
		{1, 9, 11},
		{2, 3, 7},
		{2, 5, 7},
		{2, 7, 9},
		{3, 4, 5},
		{3, 4, 9},
		{3, 5, 6},
		{3, 5, 8},
		{3, 5, 10},
		{3, 5, 11},
		{3, 6, 9}
	};
	int reg[] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
	int reg2[] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
	int g1[2046];
	//int no_nh_out[2046];
	for (int i = 0; i < 2046; i++)
	{
		g1[i] = reg[10];
		int save1 = reg[0] + reg[6] + reg[7] + reg[8] + reg[9] + reg[10];
		for (int j = 10; j > 0; j--)
		{
			reg[j] = reg[j - 1];
		}
		reg[0] = save1 % 2;
	}
	if (svnum <= 37)
	{
		int g2[2046];
		for (int i = 0; i < 2046; i++)
		{
			g2[i] = reg2[gs2[2 * svnum - 2] - 1] + reg2[gs2[2 * svnum - 1] - 1];
			g2[i] = g2[i] % 2;
			int save2 = reg2[0] + reg2[1] + reg2[2] + reg2[3] + reg2[4] + reg2[7] + reg2[8] + reg2[10];
			for (int j = 10; j > 0; j--)
			{
				reg2[j] = reg2[j - 1];
			}
			reg2[0] = save2 % 2;
		}
		for (int x = 0; x < 2046; x++)
		{
			out[x] = g1[x] + g2[x];
			out[x] = out[x] % 2;
			if (!out[x])
			{
				out[x] = -1;
			}
		}
	}
	else
	{
		int g2[2046];
		for (int i = 0; i < 2046; i++)
		{
			int sat = svnum - 37;
			g2[i] = reg2[gs2_continue[sat - 1][0] - 1] + reg2[gs2_continue[sat - 1][1] - 1] + reg2[gs2_continue[sat - 1][2] - 1];
			g2[i] = g2[i] % 2;
			int save2 = reg2[0] + reg2[1] + reg2[2] + reg2[3] + reg2[4] + reg2[7] + reg2[8] + reg2[10];
			for (int j = 10; j > 0; j--)
			{
				reg2[j] = reg2[j - 1];
			}
			reg2[0] = save2 % 2;
		}
		for (int x = 0; x < 2046; x++)
		{
			out[x] = g1[x] + g2[x];
			out[x] = out[x] % 2;
			if (!out[x])
			{
				out[x] = -1;
			}
		}
	}
	//char* str = NULL;
	//str = (char*)malloc(10);
	//FILE* feph;
	//sprintf(str, "%dCAcode.txt", svnum);
	//feph = fopen(str, "wt+");

	////写电文部分
	//for (int jj = 0; jj < 2046; jj++)
	//{
	//	fprintf(feph, "%#01d \n", no_nh_out[jj]);
	//}
	//fclose(feph);
}



//将UTM时间转换为BDS时间
void date2bds(const datetime_t* t, bdstime_t* g)
{
	int doy[12] = { 0,31,59,90,120,151,181,212,243,273,304,334 };//每月第一天对应年中的天数
	int ye;
	int de;
	int lpdays;

	ye = t->y - 2006;

	// Compute the number of leap days since Jan 5/Jan 6, 1980.
	lpdays = (ye - 2) / 4 + 1;
	if (((ye - 2) % 4) == 0 && t->m <= 2)
		lpdays--;

	// Compute the number of days elapsed since Jan 5/Jan 6, 1980.
	de = ye * 365 + doy[t->m - 1] + t->d - 1 + lpdays;

	// Convert time to GPS weeks and seconds.
	g->week = de / 7;
	g->sec = (double)((de % 7)) * SECONDS_IN_DAY + t->hh * SECONDS_IN_HOUR
		+ t->mm * SECONDS_IN_MINUTE + t->sec;

	return;
}

//将北斗时间转换为UTM时间
void bds2date(const bdstime_t* g, datetime_t* t)
{
	// Convert Julian day number to calendar date
	int c = (int)(7 * g->week + floor(g->sec / 86400.0) + 2444245.0) + 1537;
	int d = (int)((c - 122.1) / 365.25);
	int e = 365 * d + d / 4;
	int f = (int)((c - e) / 30.6001);

	t->d = c - e - (int)(30.6001 * f) - 5;
	t->m = f - 1 - 12 * (f / 14);
	t->y = d - 4715 - ((7 + t->m) / 10) + 26;

	t->hh = ((int)(g->sec / 3600.0)) % 24;
	t->mm = ((int)(g->sec / 60.0)) % 60;
	t->sec = g->sec - 60.0 * floor(g->sec / 60.0);

	return;
}

/*! \brief Convert Earth-centered Earth-fixed (ECEF) into Lat/Long/Heighth
*  \param[in] xyz Input Array of X, Y and Z ECEF coordinates
*  \param[out] llh Output Array of Latitude, Longitude and Height
*/

//地心地固坐标系转换为BLH坐标系
void xyz2llh(const double *xyz, double *llh)
{
	double a, eps, e, e2;
	double x, y, z;
	double rho2, dz, zdz, nh, slat, n, dz_new;

	a = WGS84_RADIUS;
	e = WGS84_ECCENTRICITY;

	eps = 1.0e-3;
	e2 = e*e;

	if (normVect(xyz) < eps)
	{
		// Invalid ECEF vector
		llh[0] = 0.0;
		llh[1] = 0.0;
		llh[2] = -a;

		return;
	}

	x = xyz[0];
	y = xyz[1];
	z = xyz[2];

	rho2 = x*x + y*y;
	dz = e2*z;

	while (1)
	{
		zdz = z + dz;
		nh = sqrt(rho2 + zdz*zdz);
		slat = zdz / nh;
		n = a / sqrt(1.0 - e2*slat*slat);
		dz_new = n*e2*slat;

		if (fabs(dz - dz_new) < eps)
			break;

		dz = dz_new;
	}

	llh[0] = atan2(zdz, sqrt(rho2));
	llh[1] = atan2(y, x);
	llh[2] = nh - n;

	return;
}

/*! \brief Convert Lat/Long/Height into Earth-centered Earth-fixed (ECEF)
*  \param[in] llh Input Array of Latitude, Longitude and Height
*  \param[out] xyz Output Array of X, Y and Z ECEF coordinates
*/
//BLH坐标系转换为地心地固坐标系
void llh2xyz(const double *llh, double *xyz)
{
	double n;
	double a;
	double e;
	double e2;
	double clat;
	double slat;
	double clon;
	double slon;
	double d, nph;
	double tmp;

	a = WGS84_RADIUS;
	e = WGS84_ECCENTRICITY;
	e2 = e*e;

	clat = cos(llh[0]);
	slat = sin(llh[0]);
	clon = cos(llh[1]);
	slon = sin(llh[1]);
	d = e*slat;

	n = a / sqrt(1.0 - d*d);
	nph = n + llh[2];

	tmp = nph*clat;
	xyz[0] = tmp*clon;
	xyz[1] = tmp*slon;
	xyz[2] = ((1.0 - e2)*n + llh[2])*slat;

	return;
}

/*! \brief Compute the intermediate matrix for LLH to ECEF
*  \param[in] llh Input position in Latitude-Longitude-Height format
*  \param[out] t Three-by-Three output matrix
*/
//计算LLH到ECEF的中间矩阵
void ltcmat(const double *llh, double t[3][3])
{
	double slat, clat;
	double slon, clon;

	slat = sin(llh[0]);
	clat = cos(llh[0]);
	slon = sin(llh[1]);
	clon = cos(llh[1]);

	t[0][0] = -slat*clon;
	t[0][1] = -slat*slon;
	t[0][2] = clat;
	t[1][0] = -slon;
	t[1][1] = clon;
	t[1][2] = 0.0;
	t[2][0] = clat*clon;
	t[2][1] = clat*slon;
	t[2][2] = slat;

	return;
}

/*! \brief Convert Earth-centered Earth-Fixed to ?
*  \param[in] xyz Input position as vector in ECEF format
*  \param[in] t Intermediate matrix computed by \ref ltcmat
*  \param[out] neu Output position as North-East-Up format
*/
//neu坐标系是以用户位置为原点，主要用来计算方位角和俯仰角
void ecef2neu(const double *xyz, double t[3][3], double *neu)
{
	neu[0] = t[0][0] * xyz[0] + t[0][1] * xyz[1] + t[0][2] * xyz[2];
	neu[1] = t[1][0] * xyz[0] + t[1][1] * xyz[1] + t[1][2] * xyz[2];
	neu[2] = t[2][0] * xyz[0] + t[2][1] * xyz[1] + t[2][2] * xyz[2];

	return;
}

/*! \brief Convert North-Eeast-Up to Azimuth + Elevation
*  \param[in] neu Input position in North-East-Up format
*  \param[out] azel Output array of azimuth + elevation as double
*/
//在nue坐标系下，计算方位角和俯仰角
void neu2azel(double *azel, const double *neu)
{
	double ne;

	azel[0] = atan2(neu[1], neu[0]);
	if (azel[0] < 0.0)
		azel[0] += (2.0*PI);

	ne = sqrt(neu[0] * neu[0] + neu[1] * neu[1]);
	azel[1] = atan2(neu[2], ne);

	return;
}

/*! \brief Compute Satellite position, velocity and clock at given time
*  \param[in] eph Ephemeris data of the satellite
*  \param[in] g GPS time at which position is to be computed
*  \param[out] pos Computed position (vector)
*  \param[out] vel Computed velociy (vector)
*  \param[clk] clk Computed clock
*/
//计算卫星的位置，速度，时钟同步
void satpos(ephem_BDS_t eph, bdstime_t g, double *pos, double *vel, double *clk)
{
	// Computing Satellite Velocity using the Broadcast Ephemeris
	// http://www.ngs.noaa.gov/gps-toolbox/bc_velo.htm

	double tk;
	double mk;
	double ek;
	double ekold;
	double ekdot;
	double cek, sek;
	double pk;
	double pkdot;
	double c2pk, s2pk;
	double uk;
	double ukdot;
	double cuk, suk;
	double ok;
	double sok, cok;
	double ik;
	double ikdot;
	double sik, cik;
	double rk;
	double rkdot;
	double xpk, ypk;
	double xpkdot, ypkdot;
	double miu;
	double n0;
	double n;
	double relativistic, OneMinusecosE, tmp;
	double sq1e2;
	double A;
	double omgkdot;
	tk = g.sec - eph.toe.sec;//观测历元到参考历元的时间差

	if (tk > SECONDS_IN_HALF_WEEK)//
		tk -= SECONDS_IN_WEEK;
	else if (tk < -SECONDS_IN_HALF_WEEK)//
		tk += SECONDS_IN_WEEK;
	sq1e2 = sqrt(1 - eph.ecc * eph.ecc);//sqrt(1-e^2)
	miu = 3.986004418e+14;//地心引力常数,和gps略微不同
	A = eph.sqrt_a * eph.sqrt_a;//A=(sqrt(A))^2
	n0 = sqrt(miu / (A * A * A));//n0=sqrt(niu/A^3)
	n = n0 + eph.delta_n;//n = n0+Δn
	mk = eph.m0 + n*tk;//平近点角   mk=m0+n*tk           
	omgkdot = eph.omega_dot - OMEGA_EARTH;//地球自转角速度,和gps的数值也略微不同
					   
	//这里主要就是相比gps程序在读广播星历时的处理移到了这里,可以对比一下
	//eph[ieph][sv].A = eph[ieph][sv].sqrta * eph[ieph][sv].sqrta;
	//eph[ieph][sv].n = sqrt(GM_EARTH / (eph[ieph][sv].A * eph[ieph][sv].A * eph[ieph][sv].A)) + eph[ieph][sv].deltan;
	//eph[ieph][sv].sq1e2 = sqrt(1.0 - eph[ieph][sv].ecc * eph[ieph][sv].ecc);
	//eph[ieph][sv].omgkdot = eph[ieph][sv].omgdot - OMEGA_EARTH;
	
	ek = mk;
	ekold = ek + 1.0;//

	OneMinusecosE = 0; // Suppress the uninitialized warning.
	while (fabs(ek - ekold) > 1.0E-14)//
	{
		ekold = ek;
		OneMinusecosE = 1.0 - eph.ecc*cos(ekold);
		ek = ek + (mk - ekold + eph.ecc*sin(ekold)) / OneMinusecosE;
	}

	sek = sin(ek);
	cek = cos(ek);

	ekdot = n / OneMinusecosE;

	relativistic = -4.442807309E-10*eph.ecc*eph.sqrt_a*sek;//GPS是-4.442807633E-10；F=－2*(μ^(1/2))/C^2；也稍微有所不同

	pk = atan2(sq1e2*sek, cek - eph.ecc) + eph.omega; //pk这里就是vk  aop就是omega       sqle2为sqrt(1-e^2)
	pkdot = sq1e2*ekdot / OneMinusecosE;

	s2pk = sin(2.0*pk);
	c2pk = cos(2.0*pk);

	uk = pk + eph.cus*s2pk + eph.cuc*c2pk;
	suk = sin(uk);
	cuk = cos(uk);
	ukdot = pkdot*(1.0 + 2.0*(eph.cus*c2pk - eph.cuc*s2pk));

	rk = A*OneMinusecosE + eph.crc*c2pk + eph.crs*s2pk;
	rkdot = A*eph.ecc*sek*ekdot + 2.0*pkdot*(eph.crs*c2pk - eph.crc*s2pk);

	ik = eph.i0 + eph.idot*tk + eph.cic*c2pk + eph.cis*s2pk;
	sik = sin(ik);
	cik = cos(ik);
	ikdot = eph.idot + 2.0*pkdot*(eph.cis*c2pk - eph.cic*s2pk);

	xpk = rk*cuk;//计算卫星在轨道平面内的坐标
	ypk = rk*suk;//
	xpkdot = rkdot*cuk - ypk*ukdot;
	ypkdot = rkdot*suk + xpk*ukdot;

	ok = eph.omega0 + tk* omgkdot - OMEGA_EARTH*eph.toe.sec;//计算历元升交点经度  和书上的公式不一样，这样计算出来的是在ecef坐标系下吗？
	sok = sin(ok);
	cok = cos(ok);

	pos[0] = xpk*cok - ypk*cik*sok;//计算卫星坐标
	pos[1] = xpk*sok + ypk*cik*cok;
	pos[2] = ypk*sik;

	tmp = ypkdot*cik - ypk*sik*ikdot;

	vel[0] = -omgkdot *pos[1] + xpkdot*cok - tmp*sok;//eph[ieph][sv].omgkdot = eph[ieph][sv].omgdot - OMEGA_EARTH;
	vel[1] = omgkdot *pos[0] + xpkdot*sok + tmp*cok;
	vel[2] = ypk*cik*ikdot + ypkdot*sik;

	// Satellite clock correction卫星时钟改正
	tk = g.sec - eph.toc.sec;

	if (tk > SECONDS_IN_HALF_WEEK)
		tk -= SECONDS_IN_WEEK;
	else if (tk < -SECONDS_IN_HALF_WEEK)
		tk += SECONDS_IN_WEEK;

	clk[0] = eph.sv_cb + tk*(eph.sv_cd + tk*eph.sv_cdr) + relativistic - eph.tgd1;//af0 = sv_cb;af1 = sv_cd;af2 = sv_cdr; - eph.tgd1
	clk[1] = eph.sv_cd + 2.0*tk*eph.sv_cdr;

	return;
}
/// <summary>
/// 
/// </summary>
/// <param name="src"></param>
/// <param name="mode"></param>
/// <returns></returns>
//量化
unsigned char quantify(float src, int mode)//mode用于选择量化为xbit，1bit/2bit
{
	if (mode == 2)
	{
		if (src > quantify_th)//为什么是这个数
		{
			return 0x01;//3
		}
		else if (src < quantify_th && src>0)
		{
			return 0x00;//1
		}
		else if (src < -quantify_th)
		{
			return 0x03;//-3
		}
		else if (src > -quantify_th && src < 0)
		{
			return 0x02;//-1
		}
	}
	if (mode == 1)
	{
		if (src > 0)
			return 0x01;
		else if (src < 0)
			return 0x00;
	}
	else
	{
		printf("error:(quantify) unexpected mode");
		return -10;
	}
}

//BCH编码
unsigned long BCHcode(unsigned long word)
{
	int d0, d1, d2, d3;
	int in;
	int out1;
	d3 = d2 = d1 = d0 = 0;
	for (int i = 0; i < 11; i++)
	{
		in = (word >> (10 - i)) & 0x1;
		out1 = in ^ d3;
		d3 = d2;
		d2 = d1;
		d1 = d0 ^ out1;
		d0 = out1;
	}
	unsigned long temp = ((d3 & 0x1) << 3) | ((d2 & 0x1) << 2) | ((d1 & 0x1) << 1) | (d0 & 0x1);
	unsigned long out = word << 4 | temp;
	return out;
}

//用于验证
int BCHdecode(unsigned long word)
{
	char decode = 0;
	char i, d0, d1, d2, d3;
	char in, out;
	d0 = d1 = d2 = d3 = 0;

	for (i = 0; i < 15; i++)
	{
		in = char((word >> (14 - i)) & 0x1);
		out = d3;
		d3 = d2;
		d2 = d1;
		d1 = out ^ d0;
		d0 = out ^ in;
	}
	decode = d3 | (d2 << 1) | (d1 << 2) | (d0 << 3);
	return decode;

}

//用于验证，把30位的北斗字按照奇偶分开，分成两个15位的字
void SplitBDSWord(unsigned long word0, unsigned long* wordA, unsigned long* wordB)
{
	char i, in;
	*wordA = 0;
	*wordB = 0;
	for (i = 0; i < 30; i += 2)
	{
		in = char((word0 >> i) & 0x1);
		*wordA = *wordA | (in << (i >> 1));
		in = char((word0 >> (i + 1)) & 0x1);
		*wordB = *wordB | (in << (i >> 1));
	}
}

//是SplitBDSWord函数的逆过程
void combBDSword(unsigned long wordA, unsigned long wordB, unsigned long* word)
{
	char in1, in2;
	*word = 0;
	for (int i = 0; i < 15; i++)
	{
		in1 = char((wordA >> i) & 0x1);
		in2 = char((wordB >> i) & 0x1);
		*word = *word | (in1 << (2 * i + 1)) | (in2 << (2 * i));
	}
}

/*! \brief Compute Subframe from Ephemeris
*  \param[in] eph Ephemeris of given SV
*  \param[out] sbf Array of five sub-frames, 10 long words each
*/
//根据读取到的星历的信息，产生导航电文，并添加校验码，存到sbf_results中。注意此时没有添加周内秒计数SOW
void eph2sbf(const ephem_BDS_t eph, const ionoutc_t ionoutc, unsigned long sbf_results[5][N_DWRD_SBF])
{
	unsigned long sbf[5][N_DWRD_SBF];
	unsigned long wn;
	unsigned long toe;
	unsigned long toc;
	unsigned long aode;
	unsigned long aodc;
	unsigned long ura;
	long delta_n;
	long cuc;
	long cus;
	long cic;
	long cis;
	long crc;
	long crs;
	unsigned long ecc;
	unsigned long sqrt_a;
	long m0;
	long omega0;
	long i0;
	//long aop;//
	long omega_dot;
	long idot;
	long af0;//sv_cb
	long af1;//sv_cd
	long af2;//sv_cdr
	long tgd1;
	long tgd2;
	long omega;

	unsigned long wna;//
	unsigned long toa;//

	signed long alpha0, alpha1, alpha2, alpha3;
	signed long beta0, beta1, beta2, beta3;
	signed long A0, A1;
	signed long dtls, dtlsf;
	unsigned long tot, wnt, wnlsf, dn;
	unsigned long sbf4_page18_svId = 56UL;
	unsigned long sow;

	//sow = (unsigned long)()

	//wn = 0UL;

	wn = (unsigned long)(eph.toc.week);
	ura = (unsigned long)(eph.sv_accuracy);
	alpha0 = (signed long)round(ionoutc.alpha0 / POW2_M30);
	alpha1 = (signed long)round(ionoutc.alpha1 / POW2_M27);
	alpha2 = (signed long)round(ionoutc.alpha2 / POW2_M24);
	alpha3 = (signed long)round(ionoutc.alpha3 / POW2_M24);
	beta0 = (signed long)round(ionoutc.beta0 / 2048.0);
	beta1 = (signed long)round(ionoutc.beta1 / 16384.0);
	beta2 = (signed long)round(ionoutc.beta2 / 65536.0);
	beta3 = (signed long)round(ionoutc.beta3 / 65536.0);
	aodc = (unsigned long)(eph.aodc);
	aode = (unsigned long)(eph.aode);
	toc = (unsigned long)(eph.toc.sec / 8.0);
	toe = (unsigned long)(eph.toe.sec / 8.0);
	af0 = (long)round((eph.sv_cb / POW2_M33));
	af1 = (long)round((eph.sv_cd / POW2_M50));
	af2 = (long)round((eph.sv_cdr / POW2_M66));
	sqrt_a = (unsigned long)round(eph.sqrt_a / POW2_M19);//
	ecc = (unsigned long)round(eph.ecc / POW2_M33);//
	omega = (long)round(eph.omega / POW2_M31 / PI);//w   不需要除以pi吧？
	delta_n = (long)round((eph.delta_n / POW2_M43 / PI));//n
	m0 = (long)round((eph.m0 / POW2_M31 / PI));
	omega0 = (long)round(eph.omega0 / POW2_M31 / PI);//
	omega_dot = (long)round(eph.omega_dot / POW2_M43 / PI);
	i0 = (long)round(eph.i0 / POW2_M31 / PI);//
	idot = (long)round(eph.idot / POW2_M43 / PI);
	cuc = (long)round(eph.cuc / POW2_M31);//
	cus = (long)round(eph.cus / POW2_M31);
	crc = (long)round(eph.crc / POW2_M6);
	crs = (long)round(eph.crs / POW2_M6);
	cic = (long)round(eph.cic / POW2_M31);//
	cis = (long)round(eph.cis / POW2_M31);//
	tgd1 = (long)(eph.tgd1 / 1e-10);//单位为纳秒
	tgd2 = (long)(eph.tgd2 / 1e-10 + 0.5);

	//// Subframe 1
	//sbf[0][0] = (0b11100010010 << 19) | (0b001 << 12) | (((toc >> 12) & 0xFF) << 3);  //sow和toc.sec有什么区别？
	//sbf[0][1] = (toc & 0xFFF) << 18 | (aodc << 12) | (ura << 8);
	//sbf[0][2] = (wn << 17) | ((toc >> 8 & 0x1FF) << 8);
	//sbf[0][3] = ((toc & 0xFF) << 22) | (tgd1 << 12) | (((tgd2 >> 6) & 0xF) << 8);
	//sbf[0][4] = (tgd2 & 0x3F) | (alpha0 << 16) | (alpha1 << 8);
	//sbf[0][5] = (alpha2 << 22) | (alpha3 << 14) | (((beta0 >> 2) & 0x3FF) << 8);
	//sbf[0][6] = ((beta0 & 0x3) << 28) | (beta1 << 20) | (((beta2 << 12) | (beta3 >> 4) & 0xF) << 8);
	//sbf[0][7] = (beta3 & 0xF) << 26 | (af2 << 15) | (((af0 >> 17) & 0x7F) << 8);
	//sbf[0][8] = (af0 & 0x1FFFF) | (((af1 >> 17) & 0x1F) << 8);
	//sbf[0][9] = ((af1 & 0x1FFFF) << 13) | (aode << 8);

	//// Subframe 2
	//sbf[1][0] = (0b11100010010 << 19) | (0b010 << 12) | (((toc >> 12) & 0xFF) << 3);
	//sbf[1][1] = (toc & 0xFFF) << 18 | (((delta_n >> 6) & 0x3FF) << 8);
	//sbf[1][2] = (delta_n & 0x3F) << 24 | (((cuc >> 2) & 0xFFFF) << 8);
	//sbf[1][3] = (cuc & 0x3) << 28 | (((m0 >> 12) & 0xFFFFF) << 8);
	//sbf[1][4] = (m0 & 0xFFF) << 18 | (((ecc >> 22) & 0x3FF) << 8);
	//sbf[1][5] = (ecc & 0x3FFFFF) << 8;
	//sbf[1][6] = (cus & 0x3FFFF) << 12 | (((crc >> 14) & 0xF) << 8);
	//sbf[1][7] = (crc & 0x3FFF) << 16 | (((crs >> 10) & 0xFF) << 8);
	//sbf[1][8] = (crs & 0x3FF) << 20 | (((sqrt_a >> 20) & 0xFFF) << 8);
	//sbf[1][9] = (sqrt_a & 0xFFFFF) << 10 | (((toe >> 15) & 0x3) << 8);

	//// Subframe 3
	//sbf[2][0] = (0b11100010010 << 19) | (0b011 << 12) | (((toc >> 12) & 0xFF) << 3);
	//sbf[2][1] = (toc & 0xFFF) << 18 | (((toe << 5) & 0x3FF) << 8);
	//sbf[2][2] = (toe & 0x1F) << 25 | (((i0 >> 15) & 0x1FFFF) << 8);
	//sbf[2][3] = (i0 & 0x7FFF) << 15 | (((cic >> 11) & 0x7F) << 8);
	//sbf[2][4] = (cic & 0x7FF) << 19 | (((omega_dot >> 13) & 0x7FF) << 8);
	//sbf[2][5] = (omega_dot & 0x1FFF) << 17 | (((cis >> 9) & 0x1FF) << 8);
	//sbf[2][6] = (cis & 0x1FF) << 21 | (((idot >> 1) & 0x1FFF) << 8);
	//sbf[2][7] = (idot & 0x1) << 29 | (((omega0 >> 11) & 0x1FFFFF) << 8);
	//sbf[2][8] = (omega0 & 0x7FF) << 19 | (((aop >> 21) & 0x7FF) << 8);
	//sbf[2][9] = (aop & 0x1FFFFF) << 9;

	//// Subframe 4
	//sbf[3][0] = (0b11100010010 << 19)| (0b100 << 12) | (((toc >> 12) & 0xFF) << 3);
	//sbf[3][1] = (toc & 0xFFF) << 18;
	//sbf[3][2] = 0UL;
	//sbf[3][3] = 0UL;
	//sbf[3][4] = 0UL;
	//sbf[3][5] = 0UL;
	//sbf[3][6] = 0UL;
	//sbf[3][7] = 0UL;
	//sbf[3][8] = 0UL;
	//sbf[3][9] = 0UL;

	//// Subframe 5
	//sbf[4][0] = (0b11100010010 << 19) | (0b101 << 12) | (((toc >> 12) & 0xFF) << 3);
	//sbf[4][1] = (toc & 0xFFF) << 18;
	//sbf[4][2] = 0UL;
	//sbf[4][3] = 0UL;
	//sbf[4][4] = 0UL;
	//sbf[4][5] = 0UL;
	//sbf[4][6] = 0UL;
	//sbf[4][7] = 0UL;
	//sbf[4][8] = 0UL;
	//sbf[4][9] = 0UL;
	//BCH译码前的子帧
	//Subframe 1
	sbf[0][0] = (0b11100010010 << 19) | (0b001 << 12) | (((toc >> 12) & 0xFF) << 3);
	sbf[0][1] = (toc & 0xFFF) << 10 | (aodc << 4) | ura;
	sbf[0][2] = (wn << 9) | ((toc >> 8) & 0x1FF);
	sbf[0][3] = ((toc & 0xFF) << 14) | ((tgd1 & 0x3FF) << 4) | ((tgd2 >> 6) & 0xF);
	sbf[0][4] = (tgd2 & 0x3F) << 16 | ((alpha0 & 0xFF) << 8) | (alpha1 & 0xFF);
	sbf[0][5] = ((alpha2 & 0xFF) << 14) | ((alpha3 & 0xFF) << 6) | ((beta0 >> 2) & 0x3F);
	sbf[0][6] = ((beta0 & 0x3) << 20) | ((beta1 & 0xFF) << 12) | ((beta2 & 0xFF) << 4) | ((beta3 >> 4) & 0xF);
	sbf[0][7] = (beta3 & 0xF) << 18 | ((af2 & 0x7FF) << 7) | ((af0 >> 17) & 0x7F);
	sbf[0][8] = (af0 & 0x1FFFF) << 5 | ((af1 >> 17) & 0x1F);
	sbf[0][9] = ((af1 & 0x1FFFF) << 5) | aode;

	// Subframe 2
	sbf[1][0] = (0b11100010010 << 19) | (0b010 << 12) | (((toc >> 12) & 0xFF) << 3);
	sbf[1][1] = (toc & 0xFFF) << 10 | ((delta_n >> 6) & 0x3FF);
	sbf[1][2] = (delta_n & 0x3F) << 16 | ((cuc >> 2) & 0xFFFF);
	sbf[1][3] = (cuc & 0x3) << 20 | ((m0 >> 12) & 0xFFFFF);
	sbf[1][4] = (m0 & 0xFFF) << 10 | ((ecc >> 22) & 0x3FF);
	sbf[1][5] = (ecc & 0x3FFFFF);
	sbf[1][6] = (cus & 0x3FFFF) << 4 | ((crc >> 14) & 0xF);
	sbf[1][7] = (crc & 0x3FFF) << 8 | ((crs >> 10) & 0xFF);
	sbf[1][8] = (crs & 0x3FF) << 12 | ((sqrt_a >> 20) & 0xFFF);
	sbf[1][9] = (sqrt_a & 0xFFFFF) << 2 | ((toe >> 15) & 0x3);

	// Subframe 3
	sbf[2][0] = (0b11100010010 << 19) | (0b011 << 12) | (((toc >> 12) & 0xFF) << 3);
	sbf[2][1] = (toc & 0xFFF) << 10 | ((toe >> 5) & 0x3FF);
	sbf[2][2] = (toe & 0x1F) << 17 | ((i0 >> 15) & 0x1FFFF);
	sbf[2][3] = (i0 & 0x7FFF) << 7 | ((cic >> 11) & 0x7F);
	sbf[2][4] = (cic & 0x7FF) << 11 | ((omega_dot >> 13) & 0x7FF);
	sbf[2][5] = (omega_dot & 0x1FFF) << 9 | ((cis >> 9) & 0x1FF);
	sbf[2][6] = (cis & 0x1FF) << 13 | ((idot >> 1) & 0x1FFF);
	sbf[2][7] = (idot & 0x1) << 21 | ((omega0 >> 11) & 0x1FFFFF);
	sbf[2][8] = (omega0 & 0x7FF) << 11 | ((omega >> 21) & 0x7FF);
	sbf[2][9] = (omega & 0x1FFFFF) << 1;

	// Subframe 4
	sbf[3][0] = (0b11100010010 << 19) | (0b100 << 12) | (((toc >> 12) & 0xFF) << 3);
	sbf[3][1] = (toc & 0xFFF) << 10;
	sbf[3][2] = 0UL;
	sbf[3][3] = 0UL;
	sbf[3][4] = 0UL;
	sbf[3][5] = 0UL;
	sbf[3][6] = 0UL;
	sbf[3][7] = 0UL;
	sbf[3][8] = 0UL;
	sbf[3][9] = 0UL;

	// Subframe 5
	sbf[4][0] = (0b11100010010 << 19) | (0b101 << 12) | (((toc >> 12) & 0xFF) << 3);
	sbf[4][1] = (toc & 0xFFF) << 10;
	sbf[4][2] = 0UL;
	sbf[4][3] = 0UL;
	sbf[4][4] = 0UL;
	sbf[4][5] = 0UL;
	sbf[4][6] = 0UL;
	sbf[4][7] = 0UL;
	sbf[4][8] = 0UL;
	sbf[4][9] = 0UL;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < N_DWRD_SBF; j++)
		{
			if (j == 0)
			{
				unsigned long temp = 0;
				temp = (sbf[i][j] >> 4) & 0x7FF;
				unsigned long temp1 = BCHcode(temp);
				sbf_results[i][j] = (sbf[i][j] >> 15) << 15 | temp1;

			}
			else
			{
				unsigned long wordA, wordB;
				wordA = sbf[i][j] & 0x7FF;//低11位
				wordB = (sbf[i][j] >> 11) & 0x7FF;//高11位
				unsigned long x = BCHcode(wordA);
				unsigned long y = BCHcode(wordB);
				combBDSword(y, x, &sbf_results[i][j]);
			}
		}
	}
}

/*! \brief Count number of bits set to 1
*  \param[in] v long word in whihc bits are counted
*  \returns Count of bits set to 1
*/
//计算输入的无符号长整型数中有几位是1
unsigned long countBits(unsigned long v)//计算输入的无符号长整型数中有几位是1
{
	unsigned long c;
	const int S[] = { 1, 2, 4, 8, 16 };
	const unsigned long B[] = {
		0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF };

	c = v;
	c = ((c >> S[0]) & B[0]) + (c & B[0]);
	c = ((c >> S[1]) & B[1]) + (c & B[1]);
	c = ((c >> S[2]) & B[2]) + (c & B[2]);
	c = ((c >> S[3]) & B[3]) + (c & B[3]);
	c = ((c >> S[4]) & B[4]) + (c & B[4]);

	return(c);
}

/*! \brief Compute the Checksum for one given word of a subframe
*  \param[in] source The input data
*  \param[in] nib Does this word contain non-information-bearing bits?
*  \returns Computed Checksum
*/
//添加gps校验位，没有用到此函数
unsigned long computeChecksum(unsigned long source, int nib)//为子帧添加校验位
{
	/*
	Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
	Bits 29 to  6 = Source data bits, d1, d2, ..., d24
	Bits  5 to  0 = Empty parity bits
	*/

	/*
	Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
	Bits 29 to  6 = Data bits transmitted by the SV, D1, D2, ..., D24
	Bits  5 to  0 = Computed parity bits, D25, D26, ..., D30
	*/

	/*
	1            2           3
	bit    12 3456 7890 1234 5678 9012 3456 7890
	---    -------------------------------------
	D25    11 1011 0001 1111 0011 0100 1000 0000
	D26    01 1101 1000 1111 1001 1010 0100 0000
	D27    10 1110 1100 0111 1100 1101 0000 0000
	D28    01 0111 0110 0011 1110 0110 1000 0000
	D29    10 1011 1011 0001 1111 0011 0100 0000
	D30    00 1011 0111 1010 1000 1001 1100 0000
	*/

	unsigned long bmask[6] = {
		0x3B1F3480UL, 0x1D8F9A40UL, 0x2EC7CD00UL,
		0x1763E680UL, 0x2BB1F340UL, 0x0B7A89C0UL };

	unsigned long D;
	unsigned long d = source & 0x3FFFFFC0UL;
	unsigned long D29 = (source >> 31) & 0x1UL;
	unsigned long D30 = (source >> 30) & 0x1UL;

	if (nib) // Non-information bearing bits for word 2 and 10
	{
		/*
		Solve bits 23 and 24 to presearve parity check
		with zeros in bits 29 and 30.
		*/

		if ((D30 + countBits(bmask[4] & d)) % 2)
			d ^= (0x1UL << 6);
		if ((D29 + countBits(bmask[5] & d)) % 2)
			d ^= (0x1UL << 7);
	}

	D = d;
	if (D30)
		D ^= 0x3FFFFFC0UL;

	D |= ((D29 + countBits(bmask[0] & d)) % 2) << 5;
	D |= ((D30 + countBits(bmask[1] & d)) % 2) << 4;
	D |= ((D29 + countBits(bmask[2] & d)) % 2) << 3;
	D |= ((D30 + countBits(bmask[3] & d)) % 2) << 2;
	D |= ((D30 + countBits(bmask[4] & d)) % 2) << 1;
	D |= ((D29 + countBits(bmask[5] & d)) % 2);

	D &= 0x3FFFFFFFUL;
	//D |= (source & 0xC0000000UL); // Add D29* and D30* from source data bits

	return(D);
}

/*! \brief Replace all 'E' exponential designators to 'D'
*  \param str String in which all occurrences of 'E' are replaced with *  'D'
*  \param len Length of input string in bytes
*  \returns Number of characters replaced
*/
//把D换成E
int replaceExpDesignator(char *str, int len)//把D换成E
{
	int i, n = 0;

	for (i = 0; i < len; i++)
	{
		if (str[i] == 'D')
		{
			n++;
			str[i] = 'E';
		}
	}

	return(n);
}

//计算两个BD时间的时间差，并返回
double sub_bdsTime(bdstime_t g1, bdstime_t g0)
{
	double dt;

	dt = g1.sec - g0.sec;
	dt += (double)(g1.week - g0.week) * SECONDS_IN_WEEK;

	return(dt);
}

//为输入的北斗时间延迟dt秒
bdstime_t incbdsTime(bdstime_t g0, double dt)
{
	bdstime_t g1;

	g1.week = g0.week;
	g1.sec = g0.sec + dt;

	g1.sec = round(g1.sec*1000.0) / 1000.0; // Avoid rounding error

	while (g1.sec >= SECONDS_IN_WEEK)
	{
		g1.sec -= SECONDS_IN_WEEK;
		g1.week++;
	}

	while (g1.sec < 0.0)
	{
		g1.sec += SECONDS_IN_WEEK;
		g1.week--;
	}

	return(g1);
}

/*! \brief Read Ephemersi data from the RINEX Navigation file */
/*  \param[out] eph Array of Output SV ephemeris data
*  \param[in] fname File name of the RINEX file
*  \returns Number of sets of ephemerides in the file
*/
//读取广播星历的信息，存放到eph结构体中，电离层参赛存到ionoutc中
int read_BDS_RinexNav_All(ephem_BDS_t eph[][MAX_BDS_SAT], ionoutc_t* ionoutc, const char* fname)
{
	FILE* fp;
	int ieph;

	int sv;
	char str[MAX_CHAR];
	char tmp[20];

	datetime_t t;
	bdstime_t g;
	bdstime_t g0;
	bdstime_t g00;
	double dt;
	int flags = 0x0;
	if (NULL == (fp = fopen(fname, "rt")))
		return(-1);

	// Clear valid flag
	for (ieph = 0; ieph < EPHEM_ARRAY_SIZE; ieph++)
		for (sv = 0; sv < MAX_BDS_SAT; sv++)
			eph[ieph][sv].vflg = 0;//清除可用信号的标志位
	// Read header lines先读头文件
	while (1)
	{
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		if (strncmp(str + 60, "END OF HEADER", 13) == 0)
			break;
		else if (strncmp(str + 60, "ION ALPHA", 9) == 0)
		{
			strncpy(tmp, str + 2, 12);
			tmp[12] = 0;
			ionoutc->alpha0 = atof(tmp);

			strncpy(tmp, str + 14, 12);
			tmp[12] = 0;
			ionoutc->alpha1 = atof(tmp);

			strncpy(tmp, str + 26, 12);
			tmp[12] = 0;
			ionoutc->alpha2 = atof(tmp);

			strncpy(tmp, str + 38, 12);
			tmp[12] = 0;
			ionoutc->alpha3 = atof(tmp);

			flags |= 0x1;
		}
		else if (strncmp(str + 60, "ION BETA", 8) == 0)
		{
			strncpy(tmp, str + 2, 12);
			tmp[12] = 0;
			ionoutc->beta0 = atof(tmp);

			strncpy(tmp, str + 14, 12);
			tmp[12] = 0;
			ionoutc->beta1 = atof(tmp);

			strncpy(tmp, str + 26, 12);
			tmp[12] = 0;
			ionoutc->beta2 = atof(tmp);

			strncpy(tmp, str + 38, 12);
			tmp[12] = 0;
			ionoutc->beta3 = atof(tmp);

			flags |= 0x1 << 1;
		}
		else if (strncmp(str + 60, "DELTA-UTC", 9) == 0)
		{
			strncpy(tmp, str + 3, 19);
			tmp[19] = 0;
			ionoutc->A0 = atof(tmp);

			strncpy(tmp, str + 22, 19);
			tmp[19] = 0;
			ionoutc->A1 = atof(tmp);

			strncpy(tmp, str + 41, 9);
			tmp[9] = 0;
			ionoutc->tot = atoi(tmp);

			strncpy(tmp, str + 50, 9);
			tmp[9] = 0;
			ionoutc->wnt = atoi(tmp);

			if (ionoutc->tot % 4096 == 0)
				flags |= 0x1 << 2;
		}
		else if (strncmp(str + 60, "LEAP SECONDS", 12) == 0)
		{
			strncpy(tmp, str, 6);
			tmp[6] = 0;
			ionoutc->dtls = atoi(tmp);

			flags |= 0x1 << 3;
		}
		ionoutc->vflg = FALSE;
		if (flags == 0xF)
			ionoutc->vflg = TRUE;
	}

	// Read ephemeris blocks
	g0.week = -1;
	ieph = 0;
	int sv0 = -1;
	while (1)
	{
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		// PRN
		if (strncmp(str, "C", 1) != 0)
			break;
		strncpy(tmp, str + 1, 2);
		tmp[2] = 0;
		sv = atoi(tmp) - 1;
		// EPOCH
		strncpy(tmp, str + 4, 4);
		tmp[4] = 0;
		t.y = atoi(tmp);
		strncpy(tmp, str + 9, 2);
		tmp[2] = 0;
		t.m = atoi(tmp);

		strncpy(tmp, str + 12, 2);
		tmp[2] = 0;
		t.d = atoi(tmp);

		strncpy(tmp, str + 15, 2);
		tmp[2] = 0;
		t.hh = atoi(tmp);

		strncpy(tmp, str + 18, 2);
		tmp[2] = 0;
		t.mm = atoi(tmp);

		strncpy(tmp, str + 21, 2);
		tmp[2] = 0;
		t.sec = atof(tmp);
		date2bds(&t, &g);
		if (g0.week == -1)
		{
			g0 = g;
			g00 = g; //存放第一个时刻的时间信息，方便回归初始值

		}
		if (sv0 == -1)
		{
			sv0 = sv;//sv0存放的是目前的卫星号
		}
		int dsv = sv - sv0;
		if (dsv != 0) //若卫星号发生变化，时间（ieph）重新归0，同时把g0也回到初始值
		{
			sv0 = sv;
			ieph = 0;
			g0 = g00;
		}
		// Check current time of clock
		dt = sub_bdsTime(g, g0);//通过比较g和g0的时间差来判断是否到了下一时刻
		if (dt >= SECONDS_IN_HOUR)//如果已经到了下一个星历
		{
			g0 = g;//把新的时间传给g0
			//ieph++; // a new set of ephemerides
			ieph = ieph + (dt / SECONDS_IN_HOUR);//有可能星历的时间不是连续的
		}

		// Date and time
		eph[ieph][sv].t = t;

		// SV CLK
		eph[ieph][sv].toc = g;//toc
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].sv_cb = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].sv_cd = atof(tmp);
		strncpy(tmp, str + 61, 19);
		tmp[19] = 0;
		eph[ieph][sv].sv_cdr = atof(tmp);

		// BROADCAST ORBIT - 1
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].aode = atof(tmp);
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].crs = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].delta_n = atof(tmp);
		strncpy(tmp, str + 61, 19);
		tmp[19] = 0;
		eph[ieph][sv].m0 = atof(tmp);

		// BROADCAST ORBIT - 2
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].cuc = atof(tmp);
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].ecc = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].cus = atof(tmp);
		strncpy(tmp, str + 61, 19);
		tmp[19] = 0;
		eph[ieph][sv].sqrt_a = atof(tmp);

		// BROADCAST ORBIT - 3
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].toe.sec = atof(tmp);
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].cic = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].omega0 = atof(tmp);
		strncpy(tmp, str + 61, 19);
		tmp[19] = 0;
		eph[ieph][sv].cis = atof(tmp);

		// BROADCAST ORBIT - 4
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].i0 = atof(tmp);
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].crc = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].omega = atof(tmp);
		strncpy(tmp, str + 61, 19);
		tmp[19] = 0;
		eph[ieph][sv].omega_dot = atof(tmp);

		// BROADCAST ORBIT - 5
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].idot = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].bdt_week = atof(tmp);

		// BROADCAST ORBIT - 6
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].sv_accuracy = atof(tmp);
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].sath1 = atof(tmp);
		strncpy(tmp, str + 42, 19);
		tmp[19] = 0;
		eph[ieph][sv].tgd1 = atof(tmp);
		strncpy(tmp, str + 61, 19);
		tmp[19] = 0;
		eph[ieph][sv].tgd2 = atof(tmp);

		// BROADCAST ORBIT - 7
		if (NULL == fgets(str, MAX_CHAR, fp))
			break;
		strncpy(tmp, str + 4, 19);
		tmp[19] = 0;
		eph[ieph][sv].ttom = atof(tmp);
		strncpy(tmp, str + 23, 19);
		tmp[19] = 0;
		eph[ieph][sv].aodc = atof(tmp);
		// Set valid flag
		eph[ieph][sv].vflg = 1;

		if ((sv == 60) && (ieph == 23))
		{
			break;//60课卫星全部读完，跳转
		}
		if (ieph == 23)//如果时间到了23，也应该跳转回0，并初始g0
		{
			ieph = 0;
			g0 = g00;
		}
		//// Update the working variables
		//eph[ieph][sv].A = eph[ieph][sv].sqrta * eph[ieph][sv].sqrta;
		//eph[ieph][sv].n = sqrt(GM_EARTH / (eph[ieph][sv].A * eph[ieph][sv].A * eph[ieph][sv].A)) + eph[ieph][sv].deltan;
		//eph[ieph][sv].sq1e2 = sqrt(1.0 - eph[ieph][sv].ecc * eph[ieph][sv].ecc);
		//eph[ieph][sv].omgkdot = eph[ieph][sv].omgdot - OMEGA_EARTH;
	}
	fclose(fp);



	//return(sv);
	return(24);
}

//返回电离层时延，这个没动
double ionosphericDelay(const ionoutc_t *ionoutc, bdstime_t g, double *llh, double *azel)//返回电离层时延
{
	double iono_delay = 0.0;
	double E, phi_u, lam_u, F;

	if (ionoutc->enable == FALSE)
		return (0.0); // No ionospheric delay

	E = azel[1] / PI;
	phi_u = llh[0] / PI;
	lam_u = llh[1] / PI;

	// Obliquity factor
	F = 1.0 + 16.0*pow((0.53 - E), 3.0);


	if (ionoutc->vflg == FALSE)
		iono_delay = F*5.0e-9*SPEED_OF_LIGHT;
	else
	{
		double t, psi, phi_i, lam_i, phi_m, phi_m2, phi_m3;
		double AMP, PER, X, X2, X4;

		// Earth's central angle between the user position and the earth projection of
		// ionospheric intersection point (semi-circles)
		psi = 0.0137 / (E + 0.11) - 0.022;

		// Geodetic latitude of the earth projection of the ionospheric intersection point
		// (semi-circles)
		phi_i = phi_u + psi*cos(azel[0]);
		if (phi_i > 0.416)
			phi_i = 0.416;
		else if (phi_i < -0.416)
			phi_i = -0.416;

		// Geodetic longitude of the earth projection of the ionospheric intersection point
		// (semi-circles)
		lam_i = lam_u + psi*sin(azel[0]) / cos(phi_i*PI);

		// Geomagnetic latitude of the earth projection of the ionospheric intersection
		// point (mean ionospheric height assumed 350 km) (semi-circles)
		phi_m = phi_i + 0.064*cos((lam_i - 1.617)*PI);
		phi_m2 = phi_m*phi_m;
		phi_m3 = phi_m2*phi_m;

		AMP = ionoutc->alpha0 + ionoutc->alpha1*phi_m
			+ ionoutc->alpha2*phi_m2 + ionoutc->alpha3*phi_m3;
		if (AMP < 0.0)
			AMP = 0.0;

		PER = ionoutc->beta0 + ionoutc->beta1*phi_m
			+ ionoutc->beta2*phi_m2 + ionoutc->beta3*phi_m3;
		if (PER < 72000.0)
			PER = 72000.0;

		// Local time (sec)
		t = SECONDS_IN_DAY / 2.0*lam_i + g.sec;
		while (t >= SECONDS_IN_DAY)
			t -= SECONDS_IN_DAY;
		while (t < 0)
			t += SECONDS_IN_DAY;

		// Phase (radians)
		X = 2.0*PI*(t - 50400.0) / PER;

		if (fabs(X) < 1.57)
		{
			X2 = X*X;
			X4 = X2*X2;
			iono_delay = F*(5.0e-9 + AMP*(1.0 - X2 / 2.0 + X4 / 24.0))*SPEED_OF_LIGHT;
		}
		else
			iono_delay = F*5.0e-9*SPEED_OF_LIGHT;
	}

	return (iono_delay);
}

/*! \brief Compute range between a satellite and the receiver
*  \param[out] rho The computed range
*  \param[in] eph Ephemeris data of the satellite
*  \param[in] g GPS time at time of receiving the signal
*  \param[in] xyz position of the receiver
*/
void computeRange(range_t *rho, ephem_BDS_t eph, ionoutc_t *ionoutc, bdstime_t g, double xyz[])//伪距计算、方位角、俯仰角、电离层误差
{
	double pos[3], vel[3], clk[2];
	double pos_t[3], vel_t[3], clk_t[2];
	double los[3];
	double tau;
	double range, rate;
	double xrot, yrot;

	double llh[3], neu[3];
	double tmat[3][3];
	double cw,sw;
	double speedoflight = 299792458.458;
	double tau_test;
	bdstime_t g_t=g;
	// SV position at time of the pseudorange observation.
	satpos(eph, g, pos, vel, clk);//得到卫星位置，速度

	// Receiver to satellite vector and light-time.
	subVect(los, pos, xyz);//delta    通过卫星位置和用户坐标，得到伪距
	
	tau = normVect(los) / SPEED_OF_LIGHT;//伪距除以时间，得到传输时延
	tau_test = tau;
	// Extrapolate the satellite position backwards to the transmission time.
	pos[0] -= vel[0] * tau;//根据传输时延，和卫星速度，反推卫星位置
	pos[1] -= vel[1] * tau;
	pos[2] -= vel[2] * tau;

	for (int ii = 0; ii < 5; ii++) //使用迭代的方法精度得到提高
	{
		g_t.sec = g.sec - tau_test;//得到发射时间；
		satpos(eph, g_t, pos_t, vel_t, clk_t);//得到卫星位置，速度
		cw = cos(OMEGA_EARTH * tau_test);
		sw = sin(OMEGA_EARTH * tau_test);
		pos_t[0] = cw * pos_t[0] + sw * pos_t[1];
		pos_t[1] = -sw * pos_t[0] + cw * pos_t[1];
		subVect(los, pos_t, xyz);//去除地球自转引起的误差
		tau_test = normVect(los) / speedoflight;//得到新的传播时延
	}

	// Earth rotation correction. The change in velocity can be neglected.
	xrot = pos[0] + pos[1] * OMEGA_EARTH*tau;//修正地球转动带来的影响
	yrot = pos[1] - pos[0] * OMEGA_EARTH*tau;
	pos[0] = xrot;
	pos[1] = yrot;

	// New observer to satellite vector and satellite range.
	subVect(los, pos, xyz);
	range = normVect(los);
	rho->d = range;//在接收时刻卫星和接收机的距离
	tau = range / SPEED_OF_LIGHT;//得到传输时延

	// Pseudorange.
	rho->range = range - SPEED_OF_LIGHT* clk_t[0];//去除钟差的影响

	// Relative velocity of SV and receiver.
	rate = dotProd(vel, los) / range;//卫星和接收机之间的相对速度标量

	// Pseudorange rate.伪距率
	rho->rate = rate; // - SPEED_OF_LIGHT*clk[1];

					  // Time of application.
	rho->g = g;

	// Azimuth and elevation angles.方位角和仰角
	xyz2llh(xyz, llh);
	ltcmat(llh, tmat);
	ecef2neu(los, tmat, neu);//
	neu2azel(rho->azel, neu);

	// Add ionospheric delay
	rho->iono_delay = ionosphericDelay(ionoutc, g, llh, rho->azel) / 0.6;
	rho->range += rho->iono_delay;//电离层延迟对码相位是延迟作用
	return;
}

/*! \brief Compute the code phase for a given channel (satellite)
*  \param chan Channel on which we operate (is updated)
*  \param[in] rho1 Current range, after \a dt has expired
*  \param[in dt delta-t (time difference) in seconds
*/
void computeCodePhase(channel_t *chan, range_t rho1, double dt)//码相位计算  chan是上一时刻的伪距，rho1是最新的伪距信息，计算多普勒频移需要两个时刻的伪距才能运算
{
	double ms;
	int ims;
	double rhorate;

	// Pseudorange rate.
  	rhorate = (rho1.range - chan->rho0.range) / dt;//距离差除时间差得到平均速度//得到径向速度

	// Carrier and code frequency.
	chan->f_carr = -rhorate / LAMBDA_B1C;//f_carr就是多普勒频移，由于生成的是零中频信号，载波偏移就是多普勒频移fc=-v/c*f 未考虑钟飘修正项？
	//chan->f_carr = 1000;//just for test
	chan->f_code = CODE_FREQ + chan->f_carr* B1I_CARR_TO_CODE;//得到实际码速率

	// Initial code phase and data bit counters.
	double a = chan->rho0.g.sec-chan->g0.sec;
 	ms = ((sub_bdsTime(chan->rho0.g, chan->g0) + 6.0) - chan->rho0.range / SPEED_OF_LIGHT)*1000.0;//6s是一个子帧的时间,避免减去传播时间后出现负数，算到上一子帧
	
	ims = (int)ms;
	chan->code_phase = (ms - (double)ims)*CA_SEQ_LEN; // 得到初始的码相位

	chan->iword = ims / 600; // 1 word = 30 bits = 600 ms  第几个字
	ims -= chan->iword  * 600;

	chan->ibit = ims / 20; // 1 bit = 20 code = 20 ms  第几比特
	ims -= chan->ibit * 20;

	chan->icode = ims; // 1 code = 1 ms  第几个码片

	chan->codeCA = chan->ca[(int)chan->code_phase] * 2 - 1;//把0换成-1
	//chan->dataBit = (int)((chan->dwrd[chan->iword] >> (29 - chan->ibit)) & 0x1UL) * 2 - 1;//发的是上一帧的第五子帧末尾    反正也没用到，报错就注释了

	// Save current pseudorange
	chan->rho0 = rho1;//把此刻最新的伪距信息存放到chan中



	//思路错误
	////计算B1C扩频码的码相位、B1C电文的比特位数
	//chan->b1c.f_B1cCode = B1C_CODE_FREQ + chan->f_carr * B1C_CARR_TO_CODE;
	////int temp_ms_count = (int)ms / 10;
	////int temp_ms = ms - 10 * temp_ms_count;
	//chan->b1c.B1c_CodePhase = (ms - ((int)ms/10*10)) * 1023;//B1C扩频码一组长度为10230，需要10ms发送，去除掉当前ms数十位以上的数据，然后乘以1023就是当前的码相位
	//chan->b1c.B1c_ibit = (int)ms / 10;//10ms发送1bit
	//chan->b1c.B1c_icode = (int)ms / 10;//icode记录的是距离上一bit的时刻又经过了几组完整的码片，但是对于B1C信号而言，发送一组码片和1bit电文的时间是一致的，所以发送一组主码的时间为ms数整除10向下取整。
	return;
}

/*! \brief Read the list of user motions from the input file
*  \param[out] xyz Output array of ECEF vectors for user motion
*  \param[[in] filename File name of the text input file
*  \returns Number of user data motion records read, -1 on error
*/
//读取用户轨迹
int readUserMotion(double (*xyz)[3], const char *filename,int tsim)
{
	FILE *fp;
	int numd;
	char str[MAX_CHAR];
	double t, x, y, z;
	if (NULL == (fp = fopen(filename, "rt")))
		return(-1);

	for (numd = 0; numd <tsim ; numd++)
	{
		if (fgets(str, MAX_CHAR, fp) == NULL)
			break;

		if (EOF == sscanf(str, "%lf,%lf,%lf,%lf", &t, &x, &y, &z)) // Read CSV line
			break;

		xyz[numd][0] = x;
		xyz[numd][1] = y;
		xyz[numd][2] = z;
	}

	fclose(fp);

	return (numd);
}

//读取GGA格式的用户轨迹（没用到）
int readNmeaGGA(double (*xyz)[3], const char *filename)
{
	FILE *fp;
	int numd = 0;
	char str[MAX_CHAR];
	char *token;
	double llh[3], pos[3];
	char tmp[8];

	if (NULL == (fp = fopen(filename, "rt")))
		return(-1);

	while (1)
	{
		if (fgets(str, MAX_CHAR, fp) == NULL)
			break;

		token = strtok(str, ",");

		if (strncmp(token + 3, "GGA", 3) == 0)
		{
			token = strtok(NULL, ","); // Date and time

			token = strtok(NULL, ","); // Latitude
			strncpy(tmp, token, 2);
			tmp[2] = 0;

			llh[0] = atof(tmp) + atof(token + 2) / 60.0;

			token = strtok(NULL, ","); // North or south
			if (token[0] == 'S')
				llh[0] *= -1.0;

			llh[0] /= R2D; // in radian

			token = strtok(NULL, ","); // Longitude
			strncpy(tmp, token, 3);
			tmp[3] = 0;

			llh[1] = atof(tmp) + atof(token + 3) / 60.0;

			token = strtok(NULL, ","); // East or west
			if (token[0] == 'W')
				llh[1] *= -1.0;

			llh[1] /= R2D; // in radian

			token = strtok(NULL, ","); // GPS fix
			token = strtok(NULL, ","); // Number of satellites
			token = strtok(NULL, ","); // HDOP

			token = strtok(NULL, ","); // Altitude above meas sea level

			llh[2] = atof(token);

			token = strtok(NULL, ","); // in meter

			token = strtok(NULL, ","); // Geoid height above WGS84 ellipsoid

			llh[2] += atof(token);

			// Convert geodetic position into ECEF coordinates
			llh2xyz(llh, pos);

			xyz[numd][0] = pos[0];
			xyz[numd][1] = pos[1];
			xyz[numd][2] = pos[2];

			// Update the number of track points
			numd++;

			if (numd >= USER_MOTION_SIZE)
				break;
		}
	}

	fclose(fp);

	return (numd);
}

//把子帧转换为导航电文
int generateNavMsg(bdstime_t g, channel_t *chan, int init,char * charbit)
{
	int iwrd, isbf;
	bdstime_t g0;
	unsigned long wn, tow,sow;
	unsigned long sbfwrd;
	unsigned long prevwrd;
	int nib;
	//char* str = NULL;
	//str = (char*)malloc(10);
	////写电文部分
	//FILE* feph;
	//sprintf(str, "%dEPH.txt", chan->prn);
	//feph=fopen(str, "wt+");

	g0.week = g.week;
	g0.sec = (double)(((unsigned long)(g.sec + 0.5)) / 30UL) * 30.0; //确保sec是30的整数（四舍五入） Align with the full frame length = 30 sec
	chan->g0 = g0; // Data bit reference time

	wn = (unsigned long)(g0.week % 1024);//周跳计数，目前程序运行暂不考虑会出现跳到下周的情况
	//tow = ((unsigned long)g0.sec) / 6UL;//一个子帧持续6s,一个子帧有一个tow位，所以需要除6s
	sow = (unsigned long)g0.sec;//把sec来作为第一组导航电文第一个字的周内秒计数（sow）

	if (init == 1) // Initialize subframe 5 
	{
		prevwrd = 0UL;

		for (iwrd = 0; iwrd < N_DWRD_SBF; iwrd++)//存储第五子帧的10个字，先放上一组电文的第五个子帧，这里因为是第一组导航电文，所以直接存的就是第五个子帧
		{
			//sow = sow - 6;
			sbfwrd = chan->sbf[4][iwrd];//先存入的是第五子帧

			// Add TOW-count message into HOW
			if (iwrd == 0)//为第一个字，添加SOW周内秒计数,并重新添加校验码
			{
				sbfwrd = ((sbfwrd >> 12) << 12) | (((sow >> 12) & 0xFF) << 4);//保留高18位不变，存放sow的高8位
				unsigned long temp = 0;
				temp = (sbfwrd >> 4) & 0x7FF;//去掉后4位校验位后，把低11位存档到temp中
				unsigned long temp1 = BCHcode(temp);//11位的temp进行BCH编码，得到包含4位校验码的15位
				sbfwrd = ((sbfwrd >> 15) << 15) | temp1;//把新的低15位放到sbfwrd中
			}
			if (iwrd == 1)//为第二个字，添加SOW周内秒计数，并重新添加校验码
			{
				unsigned long wordA, wordB, word;
				SplitBDSWord(sbfwrd, &wordA, &wordB);//A在后,B在前,B是第一个字,A是第二个字，将sbfwrd拆成两个15位的包含校验码的北斗字
				int x = BCHdecode(wordA);//进行BCH译码，检验是否正确
				int y = BCHdecode(wordB);
				if (x == 0 && y == 0)//译码通过
				{
					wordA = wordA >> 4;//低11位  去除校验位
					wordB = wordB >> 4;//高11位
					sbfwrd = ((wordB & 0x7ff) << 11) | (wordA & 0x7ff);//得到BCH编码前的22位的子帧
					//printf("添加sow前：\n第一个字：%03x，第二个字：%03x\n", wordB, wordA);
				}
				sbfwrd = ((sow & 0xFFF) << 10) | (sbfwrd & 0x3FF);//把sow写入编码后的子帧，保持低10位不变

				wordA = sbfwrd & 0x7FF;//低11位    重新进行BCH编码
				wordB = (sbfwrd >> 11) & 0x7FF;//高11位
				unsigned long x1 = BCHcode(wordA);
				unsigned long y1 = BCHcode(wordB);
				//printf("添加sow后：\n第一个字：%03x，第二个字：%03x\n", wordB, wordA);
				combBDSword(y1, x1, &sbfwrd);//将两个15位的字按照奇偶进行组合
			}
			

			// Compute checksum
			//感觉这一步也不是很需要啊
			//sbfwrd |= ( prevwrd << 30) & 0xC0000000UL; // 2 LSBs of the previous transmitted word 保留高两位，其余位给零，prevwrd左移30位写入sbfwrd中
			//nib = ((iwrd == 1) || (iwrd == 9)) ? 1 : 0; // Non-information bearing bits for word 2 and 10 这是什么意思
			//chan->dwrd[iwrd] = computeChecksum(sbfwrd, nib);

			//如果不是第一个或第二个字，则不需要额外的处理
			chan->dwrd[iwrd] = sbfwrd;//存放到chan中
			prevwrd = chan->dwrd[iwrd];
		}
	}
	else // Save subframe 5  这里指的是已经完成过子帧的初始化后，发射的第一个子帧是上一时刻的第五子帧
	{
		for (iwrd = 0; iwrd < N_DWRD_SBF; iwrd++)
		{
			chan->dwrd[iwrd] = chan->dwrd[N_DWRD_SBF*N_SBF + iwrd];//前50个字不需要，上一时刻的第五子帧是51-60字

			prevwrd = chan->dwrd[iwrd];
		}
		/*
		// Sanity check
		if (((chan->dwrd[1])&(0x1FFFFUL<<13)) != ((tow&0x1FFFFUL)<<13))
		{
		printf("\nWARNING: Invalid TOW in subframe 5.\n");
		return(0);
		}
		*/
	}
	//前十个字是上一帧的第五子帧，后五十个字是这一帧的五个子帧
	for (isbf = 0; isbf < N_SBF; isbf++)//每帧有5个子帧 
	{
		for (iwrd = 0; iwrd < N_DWRD_SBF; iwrd++)//每个子帧包含十个字
		{
			sbfwrd = chan->sbf[isbf][iwrd];

			// Add transmission week number to Subframe 1
			/*if ((isbf == 0) && (iwrd == 2))
				sbfwrd |= (wn & 0x3FFUL) << 20;*/
			if (iwrd == 0)//添加SOW周内秒计数,并重现添加校验码
			{
				sbfwrd = ((sbfwrd >> 12) << 12) | (((sow >> 12) & 0xFF) << 4);
				unsigned long temp = 0;
				temp = (sbfwrd >> 4) & 0x7FF;//第一个字的低11位
				unsigned long temp1 = BCHcode(temp);//第一个字的低15位
				sbfwrd = ((sbfwrd >> 15) << 15) | temp1;
			}
			if (iwrd == 1)
			{
				unsigned long wordA, wordB, word;
				SplitBDSWord(sbfwrd, &wordA, &wordB);//A在后,B在前,B是第一个字,A是第二个字
				int x = BCHdecode(wordA);
				int y = BCHdecode(wordB);
				if (x == 0 && y == 0)//译码通过
				{
					wordA = wordA >> 4;//低11位
					wordB = wordB >> 4;//高11位
					sbfwrd = ((wordB & 0x7ff) << 11) | (wordA & 0x7ff);//编码前的子帧（22位）
					//printf("添加sow前：\n第一个字：%03x，第二个字：%03x\n", wordB, wordA);
				}
				sbfwrd = ((sow & 0xFFF) << 10) | (sbfwrd & 0x3FF);//把sow写入编码后的子帧，保持低10位不变

				wordA = sbfwrd  & 0x7FF;//低11位
				wordB = (sbfwrd >> 11 )& 0x7FF;//高11位
				unsigned long x1 = BCHcode(wordA);
				unsigned long y1 = BCHcode(wordB);
				//printf("添加sow后：\n第一个字：%03x，第二个字：%03x\n", wordB, wordA);
				combBDSword(y1, x1, &sbfwrd);
			}
			//// Add TOW-count message into HOW
			//if (iwrd == 1)
			//	sbfwrd |= ((tow & 0x1FFFFUL) << 13);

			// Compute checksum
			//sbfwrd |= (prevwrd << 30) & 0xC0000000UL; // 2 LSBs of the previous transmitted word
			//nib = ((iwrd == 1) || (iwrd == 9)) ? 1 : 0; // Non-information bearing bits for word 2 and 10
			//chan->dwrd[(isbf + 1)*N_DWRD_SBF + iwrd] = computeChecksum(sbfwrd, nib);//为子帧添加校验位
			chan->dwrd[(isbf + 1) * N_DWRD_SBF + iwrd] = sbfwrd;
			prevwrd = chan->dwrd[(isbf + 1)*N_DWRD_SBF + iwrd];//子帧数乘十加当前word数
		}
		sow = sow + 6;//每一个子帧都包含一个TOW，一个子帧发送需要6s时间
	}
	////写电文部分
	//for (int jj = 0; jj < 50; jj++)
	//{
	//	if (jj % 10 == 0&&jj>0)
	//	{
	//		fprintf(feph, "\n");
	//	}
	//	fprintf(feph, "%#010x ", chan->dwrd[jj + 10]);
	//	
	//}
	//fclose(feph);

	int wrd = 0, bit = 0;
	for (int cnt = 0; cnt < 1800; cnt++,bit++)//将二进制电文转换成为char型的数组，其实应该是unsigned char
	{
		if (bit == 30)
		{
			bit = 0; wrd++;
		}
		charbit[cnt + (chan->prn - 1) * 1800] = (char)((chan->dwrd[wrd] >> (29 - bit)) & 0x1UL) * 2 - 1;
	}
	/*for (int jj = 0; jj < 60; jj++)
	{
		for (int kk = 0; kk < 30; kk++)
		{
			printf("%d ",charbit[jj * 30 + kk + (chan->prn - 1) * 1800]);
		}
		printf("\n");
	}*/
	return(1);
}

//检查卫星是否可见
int checkSatVisibility(ephem_BDS_t eph, bdstime_t g, double *xyz, double elvMask, double *azel)
{
	double llh[3], neu[3];
	double pos[3], vel[3], clk[3], los[3];
	double tmat[3][3];

	if (eph.vflg != 1)
		return (-1); // Invalid

	xyz2llh(xyz, llh);
	ltcmat(llh, tmat);

	satpos(eph, g, pos, vel, clk);
	subVect(los, pos, xyz);
	ecef2neu(los, tmat, neu);
	neu2azel(azel, neu);

	if (azel[1] * R2D > elvMask)//弧度转度数
		return (1); // Visible
					// else
	return (0); // Invisible
}
/// <summary>
/// 给可见卫星分配通道
/// </summary>
/// <param name="chan"></param>
/// <param name="eph"></param> 当前两小时星历
/// <param name="ionoutc"></param>
/// <param name="grx"></param>
/// <param name="xyz"></param>
/// <param name="elvMaskc"></param>
/// <param name="navbit"></param>
/// <returns></returns>
//为可见卫星分配通道
int allocateChannel(channel_t *chan, ephem_BDS_t*eph, ionoutc_t ionoutc, bdstime_t grx, double *xyz, double elvMaskc,char *navbit, int allocatedSat[])//
{
	int nsat = 0;
	int i, sv;
	double azel[2];

	range_t rho;
	double ref[3] = { 0.0 };
	double r_ref, r_xyz;
	double phase_ini;
	FILE* fwph = NULL;
	int n = 0;


	for (sv = 5; sv < MAX_BDS_SAT; sv++)
	{
		if (checkSatVisibility(eph[sv], grx, xyz, 10.0, azel) == 1)//首先确定卫星可见
		{
			nsat++; // Number of visible satellites  可见卫星数量

			if (allocatedSat[sv] == -1) // Visible but not allocated  =-1 卫星可见，但是未分配通道，开始为其分配通道
			{
				// Allocated new satellite.
				for (i = 0; i < MAX_CHAN; i++)//看看哪个通道尚未进行分配  
				{
					if (chan[i].prn == 0)//为i通道分配卫星
					{
						// Initialize channel
						chan[i].prn = sv + 1;
						chan[i].azel[0] = azel[0];
						chan[i].azel[1] = azel[1];
						//printf("%d\n", chan[i].prn);//打印分配通道的卫星号
						// C/A code generation
						BDB1Icodegen(chan[i].ca, chan[i].prn);

						// Generate subframe
						eph2sbf(eph[sv], ionoutc, chan[i].sbf);//产生卫星的子帧

						// Generate navigation message
						generateNavMsg(grx, &chan[i], 1,navbit);//产生导航电文

						// Initialize pseudorange 
						computeRange(&rho, eph[sv], &ionoutc, grx, xyz);//包含的伪距信息
						chan[i].rho0 = rho;

						// Initialize carrier phase
						r_xyz = rho.range;//获得伪距信息

						//computeRange(&rho, eph[sv], &ionoutc, grx, ref);//计算的是卫星到地心的距离
 						r_ref = rho.range;//

						phase_ini = (2.0*r_ref - r_xyz) / LAMBDA_B1I;//在接收时刻接收机和卫星的距离，地心原点与卫星距离
						phase_ini -= floor(phase_ini);//**********************************************
						//chan[i].carr_phase = (unsigned int)(512 * 65536.0 * phase_ini);//*65536将载波相位变为整型，*512相当于逆归一化过程
						chan[i].carr_phase = phase_ini;
						chan[i].IFcarr_phase = 0;
						// Done.
						break;
					}
				}
				// Set satellite allocation channel
				if (i < MAX_CHAN)
					allocatedSat[sv] = i;
			}
		}
		else if (allocatedSat[sv] >= 0) // Not visible but allocated
		{
			// Clear channel
			chan[allocatedSat[sv]].prn = 0;

			// Clear satellite allocation flag
			allocatedSat[sv] = -1;
		}
	}

	return(nsat);
}

void usage(void)
{
	printf("Usage: gps-sdr-sim [options]\n"
		"Options:\n"
		"  -e <gps_nav>     RINEX navigation file for GPS ephemerides (required)\n"
		"  -u <user_motion> User motion file (dynamic mode)\n"
		"  -g <nmea_gga>    NMEA GGA stream (dynamic mode)\n"
		"  -l <location>    Lat,Lon,Hgt (static mode) e.g. 30.286502,120.032669,100\n"
		"  -t <date,time>   Scenario start time YYYY/MM/DD,hh:mm:ss\n"
		"  -T <date,time>   Overwrite TOC and TOE to scenario start time\n"
		"  -d <duration>    Duration [sec] (max: %.0f)\n"
		"  -o <output>      I/Q sampling data file (default: gpssim.bin)\n"
		"  -s <frequency>   Sampling frequency [Hz] (default: 2600000)\n"
		"  -b <iq_bits>     I/Q data format [1/8/16] (default: 16)\n"
		"  -i               Disable ionospheric delay for spacecraft scenario\n"
		"  -v               Show details about simulated channels\n",
		((double)USER_MOTION_SIZE) / 10.0);

	return;
}

void zeros(int *am, int start, int endv) {
	int i, a = endv - start;
	if (a > 0) {
		for (i = 0; i < a; i++) {
			am[i] = 0;
		}
	}
}

int isEmpty(char *a) {
	if (a == NULL || strcmp(a, "") == 0)
	{
		return 1;
	}

	return 0;
}

int checkParamreq(char *navfn) {
	return !isEmpty(navfn);
}
int check_can_run(char *navfn, char *user_motion, char *nmea_gga, char *localtion, char *datetime, char *datetime_toc_toe, char *duration, char *outputfile, char *frequecy, char *iqbit) {
	int a[9] = { 0 };
	int i, dem = 0;
	if (checkParamreq(navfn))
	{
		if (!isEmpty(user_motion))a[0] = 1;
		if (!isEmpty(nmea_gga))a[1] = 1;
		if (!isEmpty(localtion))a[2] = 1;
		if (!isEmpty(datetime))a[3] = 1;
		if (!isEmpty(datetime_toc_toe))a[4] = 1;
		if (!isEmpty(duration))a[5] = 1;
		if (!isEmpty(outputfile))a[6] = 1;
		if (!isEmpty(frequecy))a[7] = 1;
		if (!isEmpty(iqbit))a[8] = 1;
		for (i = 0; i < 9; i++)
		{
			if (a[i] == 1)
				dem++;
		}
		if (dem > 0)return 1;
	}
	return 0;
}

inline//内联函数，类似于宏展开
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
	 

int bds_sim(Table *look_up,transfer_parameter *tp, cuFloatComplex* buff,size_t max_len,int freq_samp,int fc,int tsim, float* dev_i_buff)
{
	//clock_t tstart, tend, tstart1, tEnd1, sumTime;
	//char navfn[], char user_motion[], char nmea_gga[], char localtion[], char datetime[], char datetime_toc_toe[], char duration1[], char outputfile[], char frequecy[], char iqbit[]
	//char *navfn=NULL;  //D:\\PL\\GR_GPS_Cuda_mix\\GPSL1_sim_quantify\\brdc2017_0660.17n  brdc3540.14n
	//char *user_motion= NULL;        //circle1.csv
	static int count_call=0;
	char nmea_gga[256] = { 0 };
	char localtion[256] = { 0 };
	char datetime[256] = { 0 };
	char datetime_toc_toe[256] = { 0 };
	char duration1[256] = { 0 };
	char outputfile[256] = { 0 };
	char frequecy[256] = { 0 };
	char iqbit[256] = { 0 };
	char buffer[256] = { 0 };

	int bcount = 0, rcount = 0;
	int loop_count = 0;
	unsigned char ibit = 0;

	FILE *fp;
	short *a, *a1, *b, *c1, *c, *d;
	int sv;
	 static int count;//代表卫星数量，必须是一个静态变量
	static int  ieph;//代表当前选中的星历数，必须是一个静态变量
	//ephem_t eph[EPHEM_ARRAY_SIZE][MAX_SAT];//一帧30s,一共13帧，每两小时星历更改一次，13帧覆盖24h的信号
	//gpstime_t g0;

	double llh[3];

	int i;
	//channel_t chan[MAX_CHAN];
	double elvmask = 0.0; // in degree

						  //short ip,qp;
	int ip, qp;
	short *iq_buff = NULL;
	float *iq_storage = NULL;
	signed char *iq8_buff = NULL;

	//gpstime_t grx;
	double delt;
	int isamp;

	int iumd;
	int numd;
	char umfile[MAX_CHAR];
	const int const_tsim = tsim;
	//double (*xyz)[3] = NULL;
	//xyz =(double(*)[3]) malloc(sizeof(double) * tsim * 3);

	int staticLocationMode = FALSE;
	int nmeaGGA = FALSE;

	char navfile[MAX_CHAR];
	char outfile[MAX_CHAR];

	double samp_freq;
	int iq_buff_size;
	int data_format;


	float gain[MAX_CHAN];
	double path_loss;
	double ant_gain;
	double ant_pat[37];
	int ibs; // boresight angle index

	//datetime_t t0, tmin, tmax;
	//gpstime_t gmin, gmax;
	double dt;
	int igrx;

	double duration;
	int iduration;
	int verb;


	//char* navbit = NULL; //将二进制导航电文存为char型数组形式

	int timeoverwrite = FALSE; // Overwirte the TOC and TOE in the RINEX file

	//ionoutc_t ionoutc;

	////////////////////////////////////////////////////////////
	// Read options
	////////////////////////////////////////////////////////////

	// Default options
	//if (rinex_file != 0)
	//{
	//	navfn = (char*)malloc(strlen(rinex_file)+1);//如果不加1，free的时候会报错
	//	strcpy(navfn, rinex_file);
	//}
	//if (traj_file !=0)
	//{
	//	user_motion = (char*)malloc(strlen(traj_file)+1);
	//	strcpy(user_motion, traj_file);
	//}
	/*navfile[0] = 0;
	umfile[0] = 0;*/
	strcpy(outfile, "gpssim.bin");
	samp_freq = freq_samp;
	data_format = SC16;
	//g0.week = -1; // Invalid start time
	iduration = tsim;
	verb = FALSE;
	tp->ionoutc.enable = TRUE;
	/*strcpy(navfile, navfn);
	if (isEmpty(user_motion) == 0) {
		strcpy(umfile, user_motion);
		nmeaGGA = FALSE;
	}
	if (isEmpty(nmea_gga) == 0) {
		strcpy(umfile, nmea_gga);
		nmeaGGA = TRUE;
	}*/

	if (isEmpty(localtion) == 0) {
		// Static geodetic coordinates input mode
		// Added by scateu@gmail.com
		staticLocationMode = TRUE;
		sscanf(localtion, "%lf,%lf,%lf", &llh[0], &llh[1], &llh[2]);
		llh[0] = llh[0] / R2D; // convert to RAD
		llh[1] = llh[1] / R2D; // convert to RAD
	}

	if (isEmpty(outputfile) == 0) {
		strcpy(outfile, outputfile);
	}
	
	if (isEmpty(datetime_toc_toe) == 0) {
		timeoverwrite = TRUE;
		if (strncmp(datetime_toc_toe, "now", 3) == 0)
		{
			time_t timer;
			struct tm *gmt;

			time(&timer);
			gmt = gmtime(&timer);

			tp->t0.y = gmt->tm_year + 1900;
			tp->t0.m = gmt->tm_mon + 1;
			tp->t0.d = gmt->tm_mday;
			tp->t0.hh = gmt->tm_hour;
			tp->t0.mm = gmt->tm_min;
			tp->t0.sec = (double)gmt->tm_sec;

			date2bds(&(tp->t0), &(tp->g0));

		}
	}
	if (isEmpty(datetime) == 0) {
		sscanf(datetime, "%d/%d/%d,%d:%d:%lf", &tp->t0.y, &tp->t0.m, &tp->t0.d, &tp->t0.hh, &tp->t0.mm, &tp->t0.sec);
		if (tp->t0.y <= 1980 || tp->t0.m < 1 || tp->t0.m>12 || tp->t0.d < 1 || tp->t0.d>31 ||
			tp->t0.hh < 0 || tp->t0.hh>23 || tp->t0.mm < 0 || tp->t0.mm>59 || tp->t0.sec < 0.0 || tp->t0.sec >= 60.0)
		{
			printf("ERROR: Invalid date and time.\n");
			exit(1);
		}
		tp->t0.sec = floor(tp->t0.sec);
		date2bds(&(tp->t0), &(tp->g0));
	}
	if (isEmpty(duration1) == 0) {
		duration = atof(duration1);
		if (duration<0.0 || duration>((double)tsim) / 10.0)
		{
			printf("ERROR: Invalid duration.\n");
			exit(1);
		}
		iduration = (int)(duration*10.0 + 0.5);
	}


	// Buffer size	
	samp_freq = floor(samp_freq / Rev_fre);
	iq_buff_size = (int)samp_freq; // samples per 0.1sec
	samp_freq *= Rev_fre; 

	delt = 1.0 / samp_freq;

	////////////////////////////////////////////////////////////
	// Receiver position
	////////////////////////////////////////////////////////////

	numd = iduration;
	/*
	printf("xyz = %11.1f, %11.1f, %11.1f\n", xyz[0][0], xyz[0][1], xyz[0][2]);
	printf("llh = %11.6f, %11.6f, %11.1f\n", llh[0]*R2D, llh[1]*R2D, llh[2]);
	*/
	////////////////////////////////////////////////////////////
	// Read ephemeris
	////////////////////////////////////////////////////////////

	//neph = readRinexNavAll(eph, &ionoutc, navfile);  //读取rinex，返回星数？
	if (count_call == 0)
	{
		if (tp->neph == 0)
		{
			printf("ERROR: No ephemeris available.\n");
			exit(1);
		}

		if ((verb == TRUE) && (tp->ionoutc.vflg == TRUE))
		{
			printf("  %12.3e %12.3e %12.3e %12.3e\n",
				tp->ionoutc.alpha0, tp->ionoutc.alpha1, tp->ionoutc.alpha2, tp->ionoutc.alpha3);
			printf("  %12.3e %12.3e %12.3e %12.3e\n",
				tp->ionoutc.beta0, tp->ionoutc.beta1, tp->ionoutc.beta2, tp->ionoutc.beta3);
			printf("   %19.11e %19.11e  %9d %9d\n",
				tp->ionoutc.A0, tp->ionoutc.A1, tp->ionoutc.tot, tp->ionoutc.wnt);
			printf("%6d\n", tp->ionoutc.dtls);
		}

		for (sv = 0; sv < MAX_BDS_SAT; sv++)
		{
			if (tp->eph[0][sv].vflg == 1)
			{
				tp->gmin = tp->eph[0][sv].toc;
				tp->tmin = tp->eph[0][sv].t;
				break;      
			}
		}

		for (sv = 0; sv < MAX_BDS_SAT; sv++)
		{
			if (tp->eph[tp->neph - 1][sv].vflg == 1)
			{
				tp->gmax = tp->eph[tp->neph - 1][sv].toc;
				tp->tmax = tp->eph[tp->neph - 1][sv].t;         //最大运行时间,这里要求返回某颗卫星最后时刻的toc和t，那neph的含义应该不是卫星数吧？
				break;
			}
		}

		if (tp->g0.week >= 0) // Scenario start time has been set.
		{
			if (timeoverwrite == TRUE)
			{
				bdstime_t gtmp;
				datetime_t ttmp;
				double dsec;

				gtmp.week = tp->g0.week;
				gtmp.sec = (double)(((int)(tp->g0.sec)) / 7200) * 7200.0;

				dsec = sub_bdsTime(gtmp, tp->gmin);

				// Overwrite the UTC reference week number
				tp->ionoutc.wnt = gtmp.week;
				tp->ionoutc.tot = (int)gtmp.sec;

				// Iono/UTC parameters may no longer valid
				//tp->ionoutc.vflg = FALSE;

				// Overwrite the TOC and TOE to the scenario start time
				for (sv = 0; sv < MAX_BDS_SAT; sv++)
				{
					for (i = 0; i < tp->neph; i++)
					{
						if (tp->eph[i][sv].vflg == 1)
						{
							gtmp = incbdsTime(tp->eph[i][sv].toc, dsec);
							bds2date(&gtmp, &ttmp);
							tp->eph[i][sv].toc = gtmp;
							tp->eph[i][sv].t = ttmp;

							gtmp = incbdsTime(tp->eph[i][sv].toe, dsec);
							tp->eph[i][sv].toe = gtmp;
						}
					}
				}
			}
			else
			{
				if (sub_bdsTime(tp->g0, tp->gmin) < 0.0 || sub_bdsTime(tp->gmax, tp->g0) < 0.0)
				{
					printf("ERROR: Invalid start time.\n");
					printf("tmin = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n",
						tp->tmin.y, tp->tmin.m, tp->tmin.d, tp->tmin.hh, tp->tmin.mm, tp->tmin.sec,
						tp->gmin.week, tp->gmin.sec);
					printf("tmax = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n",
						tp->tmax.y, tp->tmax.m, tp->tmax.d, tp->tmax.hh, tp->tmax.mm, tp->tmax.sec,
						tp->gmax.week, tp->gmax.sec);
					exit(1);
				}
			}
		}
		else
		{
			tp->g0 = tp->gmin;     //设置当前接收机时间为gmin
			tp->t0 = tp->tmin;
		}
		printf("Start time = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n",
			tp->t0.y, tp->t0.m, tp->t0.d, tp->t0.hh, tp->t0.mm, tp->t0.sec, tp->g0.week, tp->g0.sec);
		printf("Duration = %.1f [sec]\n", ((double)numd) / 10.0);

		// Select the current set of ephemerides
		ieph = -1;

		for (i = 0; i < tp->neph; i++)
		{
			for (sv = 0; sv < MAX_BDS_SAT; sv++)
			{
				if (tp->eph[i][sv].vflg == 1)
				{
					dt = sub_bdsTime(tp->g0, tp->eph[i][sv].toc);
					if (dt >= -SECONDS_IN_HOUR && dt < SECONDS_IN_HOUR)//一个卫星轨道星历的有效时间是toe前后一小时
					{
						ieph = i;
						break;
					}
				}
			}

			if (ieph >= 0) // ieph has been set
				break;
		}

		if (ieph == -1)
		{
			printf("ERROR: No current set of ephemerides has been found.\n");
			exit(1);
		}

		////////////////////////////////////////////////////////////
		// Baseband signal buffer and output file
		////////////////////////////////////////////////////////////

		// Allocate I/Q buffer
		iq_buff = new short[2 * iq_buff_size];

		if (data_format == SC08)
		{
			iq8_buff = (signed char*)calloc(2 * iq_buff_size, 2);    //calloc(2 * iq_buff_size, 1)？
			//iq8_buff = new signed char[2 * iq_buff_size];
			if (iq8_buff == NULL)
			{
				printf("ERROR: Faild to allocate 8-bit I/Q buffer.\n");
				exit(1);
			}
		}
		else if (data_format == SC01)
		{
			iq8_buff = (signed char*)calloc(iq_buff_size / 4, 1);
			//iq8_buff = new  signed char[iq_buff_size / 4]; // byte = {I0, Q0, I1, Q1, I2, Q2, I3, Q3}
			if (iq8_buff == NULL)
			{
				printf("ERROR: Faild to allocate compressed 1-bit I/Q buffer.\n");
				exit(1);
			}
		}

		// Open output file

		////////////////////////////////////////////////////////////
		// Initialize channels
		////////////////////////////////////////////////////////////

		// Clear all channels
		for (i = 0; i < MAX_CHAN; i++)
			tp->chan[i].prn = 0;//通道失能是0，prn是从1开始的

		// Clear satellite allocation flag
		for (sv = 0; sv < MAX_BDS_SAT; sv++)
			tp->allocatedSat[sv] = -1;//卫星是否分配通道标志 default:-1

		// Initial reception time
		tp->grx = incbdsTime(tp->g0, 0.0);//初始化接收时间x,设置为和g0一样的时间

		// Allocate visible satellites 为可见星分配通道
		allocateChannel(tp->chan, tp->eph[ieph], tp->ionoutc, tp->grx, tp->xyz[0], elvmask, tp->navbit, tp->allocatedSat);



		count = 0;
		for (i = 0; i < MAX_CHAN; i++)
		{
			if (tp->chan[i].prn > 0)
			{
				printf("%02d %6.1f %5.1f %11.1f %5.1f\n", tp->chan[i].prn,
					tp->chan[i].azel[0] * R2D, tp->chan[i].azel[1] * R2D, tp->chan[i].rho0.d, tp->chan[i].rho0.iono_delay);//打印分配通道的卫星的相关信息
				count++;
			}
		}
		tp->grx = incbdsTime(tp->grx, 0.1);//接收机时间增加0.1s 每次产生0.1s数据
		////////////////////////////////////////////////////////////
		// Receiver antenna gain pattern
		////////////////////////////////////////////////////////////

		//for (i = 0; i < 37; i++)
		//	ant_pat[i] = mpow(10.0, -ant_pat_db[i] / 20.0);//对以db为单位的增益进行转换

		////////////////////////////////////////////////////////////
		// Generate baseband signals
		////////////////////////////////////////////////////////////
	}
	float* noise;
	double rr = 1000;
	for (i = 0; i < MAX_CHAN; i++)
	{
		//gain[i] = sqrt(2 * pow(10.0, (5.0 + 0.1 * i)) / samp_freq);//增益
		gain[i]= sqrt(2 * pow(10.0, (5.0)) / samp_freq);
		//gain[i] = 1;
	}

	noise = (float*)malloc(2 * iq_buff_size * sizeof(float));

	for (i = 0; i < 2 * iq_buff_size; i++)
	{
		noise[i] = (float)mgrn1(0, 1, &rr);//均值为0，方差为1
	}
	//for (i = 5e5-200; i < 5e5; i++)
	//{
	//	printf("%f %f\n", noise[2 * i], noise[2 * i + 1]);
	//}
	//int bpoint = 5;
	// Update receiver time

	//for (isamp = 0; isamp < iq_buff_size; isamp++)
	//{
	//	count = 0;
	//	for (int i = 0; i < MAX_CHAN; i++)
	//	{
	//		if (chan[i].prn > 0)
	//			count++;
 //		}

	//}

	//a  = (short*)malloc(count * iq_buff_size * sizeof(short));//13*1e6

	//b = (short*)malloc(count * iq_buff_size * sizeof(short));
	//c = (short*)malloc(count * iq_buff_size * sizeof(short));
	//c1 = (short*)malloc(count * iq_buff_size * sizeof(short));


	//point = (TPoint *)malloc(count * sizeof(TPoint));

	

	//free(CAcode);
	iq_storage = (float*)malloc(2 * sizeof(float) * iq_buff_size);
	look_up->cosTable= &cosTable512[0];
	look_up->sinTable= &sinTable512[0];
	look_up->navdata = tp->navbit;
	look_up->B1c_nav = new int[1800];
	B1C_Nav_Gen(look_up->B1c_nav);

	//for (int i = 0; i < 900* MAX_BDS_SAT; i++)
	//{
	//	look_up->navdata[2*i] = 1;
	//	look_up->navdata[2 * i + 1] = -1;
	//}


	look_up->NH = &NH[0];
	//GPSL1table.qua_buff = (int*)malloc(sizeof(int) * iq_buff_size / 5);
	GPUMemoryInit(look_up,count);//把正弦、余弦、CA码、NH码导航电文绑定纹理内存（运行效率更高）
	//申请核函数中使用的变量
	int  * dev_sum, * sum;
	double* parameters, * dev_parameters;
	float* dev_noise;
	double* para_doub, * dev_para_doub;
	//checkCuda(cudaHostAlloc((void**)&dev_buff, iq_buff_size * sizeof(float) * 2, cudaHostAllocDefault));//
	checkCuda(cudaHostAlloc((void**)&dev_noise, iq_buff_size * sizeof(float) * 2, cudaHostAllocDefault));//开辟空间
	//checkCuda(cudaMalloc((void**)&dev_i_buff, iq_buff_size * sizeof(int) * 2));
	//checkCuda(cudaHostAlloc((void**)&dev_q_buff, samples * sizeof(int), cudaHostAllocDefault));
	checkCuda(cudaMalloc((void**)&dev_sum, 2 * sizeof(int)));
	checkCuda(cudaMalloc((void**)&dev_parameters, p_n * count * sizeof(double)));
	checkCuda(cudaMemcpy(dev_noise, noise, sizeof(float) * iq_buff_size * 2, cudaMemcpyHostToDevice));
	checkCuda(cudaMalloc((void**)&dev_para_doub, pd_n * count * sizeof(double)));
	parameters = (double*)malloc(sizeof(double) * count * p_n);
	para_doub = (double*)malloc(sizeof(double) * count * pd_n);
	sum = (int*)malloc(sizeof(int) * 2);
	clock_t tstart = clock();

	//FILE* fpw;
	//fpw = fopen("..\\motion.txt", "w+");
	//fprintf(fpw, "北斗时    x   y   z\n");

	//初始化B1C信号的各项参数，包括码相位、载波相位和各种计数器
	//B1C选择的可见性与B1I信号保持一致
	//如果B1C信号和B1I信号采用相同的发射时间和相同的载波多普勒的话，由于二者帧结构的不同，B1C信号难以继续得到相应时刻的比特计数和码计数（尤其是二级子码计数器）
	//暂时没想到怎么算，所以对于B1C信号，手动设置一个多普勒值以方便下面计算
	for (int i = 0; i < MAX_CHAN; i++)
	{
		if (tp->chan[i].prn > 0)
		{
			//double dopple = 1000;
			//tp->chan[i].b1c.B1c_CarrPhase = 0;
			tp->chan[i].b1c.B1c_CodePhase = 0;
			tp->chan[i].f_SUBcarr= 14 * 1.023e6;
			tp->chan[i].IFcarr_phase = 0;
			tp->chan[i].b1c.B1c_ibit = 0;
			tp->chan[i].b1c.B1c_icode = 0;
			tp->chan[i].b1c.fa_phase = 0;
			tp->chan[i].b1c.fb_phase = 0;
			//tp->chan[i].b1c.B1c_CarrStep = (dopple + fc) / samp_freq;
			//tp->chan[i].b1c.B1c_CodeStep = (1.023e6 + 1.023e6 * dopple / CARR_FREQ) / samp_freq;
			//tp->chan[i].IFcarr_step = (tp->chan[i].f_SUBcarr + tp->chan[i].f_SUBcarr * dopple / CARR_FREQ) / samp_freq;
			//tp->chan[i].b1c.fa_step = (1.023e6 + 1.023e6 * dopple / CARR_FREQ) / samp_freq;
			//tp->chan[i].b1c.fb_step = (1.023e6 * 6 + 1.023e6 * 6 * dopple / CARR_FREQ) / samp_freq;
		}
	}





	float temp = 0;
	int start;
	if (count_call == 0)//最开始的0.1s没有产生采样点
	{
		start = 1;
	}
	else
	{
		start = 0;
	}
	for (iumd = start; iumd < numd ; iumd++)           //开始每历元，每个通道的循环   numd为simu_time
	{
		printf("\nlap = %d", iumd+ count_call* numd);
		loop_count += 1;
		for (i = 0; i < MAX_CHAN; i++)//每次都是重新计算码相位载波相位和步进
		{
			if (tp->chan[i].prn > 0)
			{
				// Refresh code phase and data bit counters
				range_t rho;
				sv = tp->chan[i].prn - 1;
				// Current pseudorange
				computeRange(&rho, tp->eph[ieph][sv], &tp->ionoutc, tp->grx, tp->xyz[iumd + count_call * numd]);//在grx时刻下接收机和卫星距离
				////打印本地北斗时时刻对应的接收机位置
				//if (i == 0)
				//{
				//	fprintf(fpw, "%f   %f   %f   %f\n", tp->grx.sec, tp->xyz[iumd + count_call * numd][0], tp->xyz[iumd + count_call * numd][1], tp->xyz[iumd + count_call * numd][2]);
				//}
				tp->chan[i].azel[0] = rho.azel[0];
				tp->chan[i].azel[1] = rho.azel[1];
				// Update code phase and data bit counters
				computeCodePhase(&(tp->chan[i]), rho, 0.1);
				if (count_call == 0 && iumd == 1)
				{
					tp->chan[i].B1I_carr = tp->chan[i].carr_phase;
				}
				//tp->chan[i].B1I_carr = tp->chan[i].carr_phase;
				while (tp->chan[i].B1I_carr < 0)
				{
					tp->chan[i].B1I_carr++;//载波相位必须在0-1之间（单位2pi）
				}
				while (tp->chan[i].B1I_carr > 1)
				{
					tp->chan[i].B1I_carr--;
				}
				//chan[i].carr_phasestep = (int)(512 * 65536.0 * (chan[i].f_carr+Fc) * delt);//需要与65536相乘吗？	
				tp->chan[i].carr_phasestep = (tp->chan[i].f_carr + fc + fc * tp->chan[i].f_carr / CARR_FREQ) * delt;//增加了中频   计算载波相位和码相位的步进
				tp->chan[i].code_phasestep = tp->chan[i].f_code / samp_freq;

				tp->chan[i].b1c.B1c_CarrStep = (tp->chan[i].f_carr + fc + fc * tp->chan[i].f_carr / CARR_FREQ) / samp_freq;
				tp->chan[i].b1c.B1c_CodeStep = (1.023e6 + 1.023e6 * tp->chan[i].f_carr / CARR_FREQ) / samp_freq;
				tp->chan[i].IFcarr_step = (tp->chan[i].f_SUBcarr + tp->chan[i].f_SUBcarr * tp->chan[i].f_carr / CARR_FREQ) / samp_freq;

				//多个载波分别叠加查表，导致误差叠加最终使得生成的载波不连续，这里载波和步进选择统一计算
				tp->chan[i].B1I_carrstep = tp->chan[i].carr_phasestep - tp->chan[i].IFcarr_step;
				while(tp->chan[i].B1I_carrstep < 0)
				{
					tp->chan[i].B1I_carrstep++;//载波步进必须为正
				}


				tp->chan[i].b1c.fa_step = (1.023e6 + 1.023e6 * tp->chan[i].f_carr / CARR_FREQ) / samp_freq;
				tp->chan[i].b1c.fb_step = (1.023e6 * 6 + 1.023e6 * 6 * tp->chan[i].f_carr / CARR_FREQ) / samp_freq;
				

			 //   tp->chan[i].b1c.B1c_CodePhase += tp->chan[i].b1c.B1c_CodeStep * iq_buff_size;
				//while (tp->chan[i].b1c.B1c_CodePhase >= 10230)
				//{
				//	tp->chan[i].b1c.B1c_CodePhase = tp->chan[i].b1c.B1c_CodePhase - 10230;
				//}
				//tp->chan[i].b1c.B1c_CarrPhase += tp->chan[i].b1c.B1c_CarrStep * iq_buff_size;
				//tp->chan[i].b1c.B1c_CarrPhase = tp->chan[i].b1c.B1c_CarrPhase - (int)tp->chan[i].b1c.B1c_CarrPhase;
				//
				//tp->chan[i].b1c.fa_phase += tp->chan[i].b1c.fa_step * iq_buff_size;
				//tp->chan[i].b1c.fa_phase = tp->chan[i].b1c.fa_phase - (int)tp->chan[i].b1c.fa_phase;

				//tp->chan[i].b1c.fb_phase += tp->chan[i].b1c.fb_step * iq_buff_size;
				//tp->chan[i].b1c.fb_phase = tp->chan[i].b1c.fb_phase - (int)tp->chan[i].b1c.fb_phase;
				//
				//tp->chan[i].IFcarr_phase += tp->chan[i].IFcarr_step * iq_buff_size;
				//tp->chan[i].b1c.B1c_ibit = (tp->chan[i].b1c.B1c_ibit + (int)(tp->chan[i].b1c.B1c_CodeStep * iq_buff_size)) % 1800;
				//tp->chan[i].b1c.B1c_icode = tp->chan[i].b1c.B1c_ibit;//icode 二级扩频码计数器，范围是0-1800恰好和bit计数器相同

				//else
				//{
				//	tp->chan[i].b1c.B1c_CodePhase = (iumd + 9 + 10 * (count_call - 1)) * tp->chan[i].b1c.B1c_CodeStep * iq_buff_size;
				//	while (tp->chan[i].b1c.B1c_CodePhase >= 10230)
				//	{
				//		tp->chan[i].b1c.B1c_CodePhase = tp->chan[i].b1c.B1c_CodePhase - 10230;
				//	}
				//	tp->chan[i].b1c.B1c_CarrPhase = (iumd + 9 + 10 * (count_call - 1)) * tp->chan[i].b1c.B1c_CarrStep * iq_buff_size;
				//	tp->chan[i].b1c.B1c_CarrPhase = tp->chan[i].b1c.B1c_CarrPhase - (int)tp->chan[i].b1c.B1c_CarrPhase;

				//	tp->chan[i].b1c.fa_phase = (iumd + 9 + 10 * (count_call - 1)) * tp->chan[i].b1c.fa_step * iq_buff_size;
				//	tp->chan[i].b1c.fa_phase = tp->chan[i].b1c.fa_phase - (int)tp->chan[i].b1c.fa_phase;

				//	tp->chan[i].b1c.fb_phase = (iumd + 9 + 10 * (count_call - 1)) * tp->chan[i].b1c.fb_step * iq_buff_size;
				//	tp->chan[i].b1c.fb_phase = tp->chan[i].b1c.fb_phase - (int)tp->chan[i].b1c.fb_phase;

				//	tp->chan[i].IFcarr_phase = (iumd + 9 + 10 * (count_call - 1)) * tp->chan[i].IFcarr_step * iq_buff_size;
				//	tp->chan[i].b1c.B1c_ibit = (int)((iumd + 9 + 10 * (count_call - 1)) * tp->chan[i].b1c.B1c_CodeStep * iq_buff_size) % 1800;
				//	tp->chan[i].b1c.B1c_icode = tp->chan[i].b1c.B1c_ibit % 20;
				//}
				// Path loss
				path_loss = 20200000.0 / rho.d;//20200000 为轨道高度 rho.d    没用到
				// Receiver antenna gain接收机天线增益
				ibs = (int)((90.0 - rho.azel[1] * R2D) / 5.0); // *****************************covert elevation to boresight  没用到
				//ant_gain = ant_pat[ibs];//天线幅值增益和仰角成反比，仰角为90度对应boresight为0度 需要根据信噪比设置
				// Signal gain
				//gain[i] = (int)(path_loss*ant_gain*100.0); // scaled by 100 天线增益与距离和仰角都有关系   这边是利用仰角来计算信号的增益
				tp->chan[i].amp = gain[i];//增益
			}
		}
		produce_samples_withCuda(look_up, &(tp->chan[0]), samp_freq,parameters, dev_parameters, sum, dev_sum, dev_i_buff, count,dev_noise, para_doub,dev_para_doub);//进行BPSK调制
		
		for (int satid = 0; satid < MAX_CHAN; satid++)//更新载波相位和子载波相位
		{
			tp->chan[satid].carr_phase += tp->chan[satid].carr_phasestep * iq_buff_size;
			tp->chan[satid].carr_phase -= int(tp->chan[satid].carr_phase);

			tp->chan[satid].IFcarr_phase += tp->chan[satid].IFcarr_step * iq_buff_size;
			tp->chan[satid].IFcarr_phase -= int(tp->chan[satid].IFcarr_phase);


			tp->chan[satid].b1c.B1c_CodePhase += tp->chan[satid].b1c.B1c_CodeStep * iq_buff_size;
			while (tp->chan[satid].b1c.B1c_CodePhase >= 10230)
			{
				tp->chan[satid].b1c.B1c_CodePhase = tp->chan[satid].b1c.B1c_CodePhase - 10230;
				tp->chan[satid].b1c.B1c_icode+=1;
				tp->chan[satid].b1c.B1c_ibit+=1;
			}
			//double temp = tp->chan[satid].b1c.B1c_CodePhase;
			//while (temp >= 1023)
			//{
			//	tp->chan[satid].b1c.B1c_icode++;
			//	tp->chan[satid].b1c.B1c_ibit++;
			//	temp = temp - 1023;
			//}
			//tp->chan[satid].b1c.B1c_CarrPhase += tp->chan[satid].b1c.B1c_CarrStep * iq_buff_size;
			//tp->chan[satid].b1c.B1c_CarrPhase = tp->chan[satid].b1c.B1c_CarrPhase - (int)tp->chan[satid].b1c.B1c_CarrPhase;

			tp->chan[satid].B1I_carr += tp->chan[satid].B1I_carrstep * iq_buff_size;
			tp->chan[satid].B1I_carr = tp->chan[satid].B1I_carr - (int)tp->chan[satid].B1I_carr;

			tp->chan[satid].b1c.fa_phase += tp->chan[satid].b1c.fa_step * iq_buff_size;
			tp->chan[satid].b1c.fa_phase = tp->chan[satid].b1c.fa_phase - (int)tp->chan[satid].b1c.fa_phase;

			tp->chan[satid].b1c.fb_phase += tp->chan[satid].b1c.fb_step * iq_buff_size;
			tp->chan[satid].b1c.fb_phase = tp->chan[satid].b1c.fb_phase - (int)tp->chan[satid].b1c.fb_phase;
		}
			//tp->chan[satid].IFcarr_phase += tp->chan[satid].IFcarr_step * iq_buff_size;
		//	tp->chan[satid].b1c.B1c_ibit = (tp->chan[satid].b1c.B1c_ibit + (int)(tp->chan[satid].b1c.B1c_CodeStep * iq_buff_size)) % 1800;
		//	tp->chan[satid].b1c.B1c_icode = tp->chan[satid].b1c.B1c_ibit;//icode 二级扩频码计数器，范围是0-1800恰好和bit计数器相同
		//}

		//for (int satid = 0; satid < MAX_CHAN; satid++)
		//{
		//	chan[satid].code_phase += chan[satid].code_phasestep * iq_buff_size;
		//	chan[satid].code_phase =(int)chan[satid].code_phase%1023+ chan[satid].code_phase- (int)chan[satid].code_phase;
		//}
		//tEnd1 = clock();
		
		if (count_call == 0)
		{
			for (int i = 0; i < iq_buff_size; i++)
			{
				buff[i + (iumd - 1) * iq_buff_size].x = look_up->buff[i].x;
				buff[i + (iumd - 1) * iq_buff_size].y = look_up->buff[i].y;
				//int temp = i + (iumd - 1) * iq_buff_size;
				//if (i == 1213-1)
				//{
				//	printf("\ni==%d    buff=%f+i*%f\n", temp, buff[temp].x, buff[temp].y);
				//}
			}
		}
		else
		{
			for (int i = 0; i < iq_buff_size; i++)
			{
				buff[i + iumd * iq_buff_size].x = look_up->buff[i].x;
				buff[i + iumd * iq_buff_size].y = look_up->buff[i].y;
			}
		}

		//for (int jj = 0; jj < iq_buff_size; jj++)
		//{
		//	//iq_storage[jj * 2] = GPSL1table.i_buff[jj];
		//	//iq_storage[jj * 2+1] = GPSL1table.i_buff[jj+iq_buff_size];
		//	buff->real(look_up->i_buff[jj]);
		//	buff->imag(look_up->i_buff[jj + iq_buff_size]);//把两路信号依次交替存放
		//	buff++;
		//}

		//for (int jj = 0; jj < 1000; jj++)
		//{
		//	//fprintf(fpw, "%.8f\n", GPSL1table.i_buff[jj]);
		//	printf("%.8f\n", GPSL1table.i_buff[jj]);
		//}
		//fclose(fpw);
 		if (data_format == SC01)//量化没用到
		{
			for (isamp = 0; isamp < 2 * iq_buff_size; isamp++)
			{
				if (isamp % 8 == 0)
					iq8_buff[isamp / 8] = 0x00;

				iq8_buff[isamp / 8] |= (iq_buff[isamp] > 0 ? 0x01 : 0x00) << (7 - isamp % 8);
			}

			//fwrite(iq8_buff, 1, iq_buff_size / 4, fp);
		}
		else if (data_format == SC08)
		{
			for (isamp = 0; isamp < 2 * iq_buff_size; isamp++)
				iq8_buff[isamp] = iq_buff[isamp] >> 4; // 12-bit bladeRF -> 8-bit HackRF

			//fwrite(iq8_buff, 1, 2 * iq_buff_size, fp);
		}
		else // data_format==SC16
		{
			
			
			/*fwrite(GPSL1table.qua_buff, 4, iq_buff_size / 5, fp);*/
		}

		//
		// Update navigation message and channel allocation every 30 seconds******每30s就要更新一次
		//

		igrx = (int)(tp->grx.sec*10.0 + 0.5);//grx.sec在乘10后很可能是.099999，需要+0.5再取整，否则可能减少了一个历元的数据

		if (igrx % 300 == 0) // Every 30 seconds 每30s更新一次
		{
			// Update navigation message
			for (i = 0; i < MAX_CHAN; i++)
			{
				if (tp->chan[i].prn > 0)
					generateNavMsg(tp->grx, &(tp->chan[i]), 0,tp->navbit);//将下一帧的数据生成导航电文，主要是更新tow，轨道参数并不会产生变化，在一个Rinex观测周期之后，轨道参数才会发生变化，即ieph++
			}
			navdata_update(look_up);//将导航电文绑定为纹理内存
			// Refresh ephemeris and subframes
			// Quick and dirty fix. Need more elegant way.
			for (sv = 0; sv < MAX_BDS_SAT; sv++)
			{
				if (tp->eph[ieph + 1][sv].vflg == 1)//
				{
					dt = sub_bdsTime(tp->eph[ieph + 1][sv].toc, tp->grx);
					if (dt < SECONDS_IN_HALF_HOUR)//小于半小时的时候开始播发下一阶段的导航电文  其实不会仿真那么久，这个if语句不会进来
					{
						ieph++;//

						for (i = 0; i < MAX_CHAN; i++)
						{
							// Generate new subframes if allocated
							if (tp->chan[i].prn != 0)
								eph2sbf(tp->eph[ieph][tp->chan[i].prn - 1], tp->ionoutc, tp->chan[i].sbf);//更新新的一帧 4，5子帧可能会换页
						}
					}

					break;
				}
			}

			// Update channel allocation
			allocateChannel(tp->chan, tp->eph[ieph], tp->ionoutc, tp->grx, tp->xyz[iumd + count_call * 10], elvmask,tp->navbit,tp->allocatedSat);//再次分配通道，30s后可能有的星已经不可见

			// Show ditails about simulated channels
			if (verb == TRUE)  //打印仿真的细节
			{
				printf("\n");
				for (i = 0; i < MAX_CHAN; i++)
				{
					if (tp->chan[i].prn > 0)
					{
						/*point[i].x = chan[i].azel[0] * R2D;
						point[i].y = chan[i].azel[1] * R2D;*/
						printf("%02d %6.1f %5.1f %11.1f %5.1f\n", tp->chan[i].prn,
							tp->chan[i].azel[0] * R2D, tp->chan[i].azel[1] * R2D, tp->chan[i].rho0.d, tp->chan[i].rho0.iono_delay);

					}
				}

			}
		}

		// Update receiver time
		tp->grx = incbdsTime(tp->grx, 0.1);//时间前进0.1s

		// Update time counter
		printf("\rRunning time = %4.1f", sub_bdsTime(tp->grx, tp->g0));

		fflush(stdout);
	}

	count_call += 1;
	GPUMemroy_delete(look_up);
	free(noise);
	checkCuda(cudaFreeHost(dev_noise));
	checkCuda(cudaFree(dev_parameters));
	checkCuda(cudaFree(dev_para_doub));

	checkCuda(cudaFree(dev_sum));
	//checkCuda(cudaFreeHost(dev_buff));
	free(parameters);
	free(para_doub);
	free(sum);
	//fclose(fpw);
	//free(a1);
//	free(c);
//	free(c1);
	//free(d);
	clock_t tend = clock();

	printf("\nDone!\n");


	// Free I/Q buffer
	delete[]iq_buff;
	free(iq_storage);

	free(iq8_buff);

	// Process time
	printf("Process time = %.1f [sec]\n", (double)(tend-tstart)/CLOCKS_PER_SEC);
	//printf("Process time = %.1f [sec]\n", sumTime);
	return iq_buff_size* loop_count;

}



