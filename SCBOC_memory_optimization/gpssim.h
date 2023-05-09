#ifndef BDSSIM_H
#define BDSSIM_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <complex>
#include "B1cCodeNav.h"
#include "cuComplex.h"
#define TRUE	(1)
#define FALSE	(0)

/*! \brief Maximum length of a line in a text file (RINEX, motion) */
#define MAX_CHAR (100)

/*! \brief Maximum number of satellites in RINEX file */
//#define MAX_SAT (32)

#define MAX_BDS_SAT (60)

/*! \brief Maximum number of channels we simulate */
#define MAX_CHAN (8)//16

/*! \brief Maximum number of user motion points */
#define USER_MOTION_SIZE (100) // max duration at 10Hz

/*! \brief Maximum duration for static mode*/
#define STATIC_MAX_DURATION (86400) // second

/*! \brief Number of subframes */
#define N_SBF (5) // 5 subframes per frame

/*! \brief Number of words per subframe */
#define N_DWRD_SBF (10) // 10 word per subframe

/*! \brief Number of words */
#define N_DWRD ((N_SBF+1)*N_DWRD_SBF) // Subframe word buffer size

/*! \brief C/A code sequence length */
#define CA_SEQ_LEN (2046)

#define SECONDS_IN_WEEK 604800.0
#define SECONDS_IN_HALF_WEEK 302400.0
#define SECONDS_IN_DAY 86400.0
#define SECONDS_IN_HALF_HOUR 1800.0
#define SECONDS_IN_HOUR 3600.0
#define SECONDS_IN_MINUTE 60.0

#define POW2_M5  0.03125
#define POW2_M6  0.015625
#define POW2_M19 1.907348632812500e-6
#define POW2_M29 1.862645149230957e-9
#define POW2_M31 4.656612873077393e-10
#define POW2_M33 1.164153218269348e-10
#define POW2_M43 1.136868377216160e-13
#define POW2_M55 2.775557561562891e-17

#define POW2_M50 8.881784197001252e-016
#define POW2_M30 9.313225746154785e-010
#define POW2_M27 7.450580596923828e-009
#define POW2_M24 5.960464477539063e-008
#define POW2_M66 1.3552527156068805425093160010874e-20

// Conventional values employed in GPS ephemeris model (ICD-GPS-200)
#define GM_EARTH 3.986005e14
//#define OMEGA_EARTH 7.2921151467e-5
//地球自转角速度
#define OMEGA_EARTH 7.2921150e-5
#define PI 3.1415926535898

#define WGS84_RADIUS	6378137.0
#define WGS84_ECCENTRICITY 0.0818191908426

#define R2D 57.2957795131

#define SPEED_OF_LIGHT 2.99792458e8
//#define LAMBDA_L1 0.190293672798365

#define LAMBDA_B1I 0.192039486310276
#define LAMBDA_B1C 0.190293672798365


/*! \brief GPS L1 Carrier frequency */
#define CARR_FREQ  (1575.42e6)
/*! \brief C/A code frequency */
#define CODE_FREQ (2.046e6)      
#define B1C_CODE_FREQ 1.023e6
//#define CARR_TO_CODE (1.0/1540.0)

#define B1I_CARR_TO_CODE (0.0012987012987012987012987012987013)//2.046/1561.098=0.001310615989515
#define B1C_CARR_TO_CODE (0.00064935064935064935064935064935065)

// Sampling data format
#define SC01 (1)
#define SC08 (8)
#define SC16 (16)

#define EPHEM_ARRAY_SIZE (24) // for daily BDS broadcast ephemers file (brdc)

//#define Freq_sample 20e6
#define Rev_fre 10
#define threadPerBlock 16 //供生成采样点的核函数使用，note that: threadPerBlock*satnum<1024
//#define Fc -6580000 //卫星中频 -(1582000000-1575420000)
#define quantify_th 0.910
#define p_n 7
#define pd_n 14

/*! \brief Structure representing BDS time */
typedef struct
{
	int week;	/*!< BDS week number (since 2006.1.1) */
	double sec; 	/*!< second inside the GPS \a week */
} bdstime_t;

/*! \brief Structure repreenting UTC time */
typedef struct
{
	int y; 		/*!< Calendar year */
	int m;		/*!< Calendar month */
	int d;		/*!< Calendar day */
	int hh;		/*!< Calendar hour */
	int mm;		/*!< Calendar minutes */
	double sec;	/*!< Calendar seconds */
} datetime_t;

typedef struct B1c_code
{
	//int datacode[10230 * MAX_BDS_SAT];
	//int pilotcode[10230 * MAX_BDS_SAT];
	//int secondcode[1800 * MAX_BDS_SAT];
	int* datacode;
	int* pilotcode;
	int* secondcode;

	int* dev_datacode;
	int* dev_pilotcode;
	int* dev_secondcode;
}; B1c_code;


typedef struct Init_Table
{
	int* CAcode;
	B1c_code B1cCode;
	int* sinTable;
	int* cosTable;
	char* navdata;
	int* B1c_nav;
	float *i_buff, * q_buff;
	cuFloatComplex* buff;
	int* qua_buff;
	float* dev_i, * dev_q;
	cudaArray* cu_cosTable;
	cudaArray* cu_sinTable;
	int* dev_CAcode;
	char* dev_navdata;
	char* dev_B1c_nav;
	int* NH;
	cudaArray* dev_NH;
}Table;
/*! \brief Structure representing ephemeris of a single satellite */
//typedef struct
//{
//	int vflg;	/*!< Valid Flag */
//	datetime_t t;
//	gpstime_t toc;	/*!< Time of Clock */
//	gpstime_t toe;	/*!< Time of Ephemeris */
//	int iodc;	/*!< Issue of Data, Clock */
//	int iode;	/*!< Isuse of Data, Ephemeris */
//	double deltan;	/*!< Delta-N (radians/sec) */
//	double cuc;	/*!< Cuc (radians) */
//	double cus;	/*!< Cus (radians) */
//	double cic;	/*!< Correction to inclination cos (radians) */
//	double cis;	/*!< Correction to inclination sin (radians) */
//	double crc;	/*!< Correction to radius cos (meters) */
//	double crs;	/*!< Correction to radius sin (meters) */
//	double ecc;	/*!< e Eccentricity */
//	double sqrta;	/*!< sqrt(A) (sqrt(m)) */
//	double m0;	/*!< Mean anamoly (radians) */
//	double omg0;	/*!< Longitude of the ascending node (radians) */
//	double inc0;	/*!< Inclination (radians) */
//	double aop;//看一下是什么，是赋值吗？
//	double omgdot;	/*!< Omega dot (radians/s) */
//	double idot;	/*!< IDOT (radians/s) */
//	double af0;	/*!< Clock offset (seconds) */
//	double af1;	/*!< rate (sec/sec) */
//	double af2;	/*!< acceleration (sec/sec^2) */
//	double tgd;	/*!< Group delay L2 bias */
//	int svhlth;
//	int codeL2;
//	// Working variables follow
//	double n; 	/*!< Mean motion (Average angular velocity) */
//	double sq1e2;	/*!< sqrt(1-e^2) */
//	double A;	/*!< Semi-major axis */
//	double omgkdot; /*!< OmegaDot-OmegaEdot */
//} ephem_t;
typedef struct
{
	int vflg;	/*!< Valid Flag */
	datetime_t t;
	bdstime_t toc;	/*!< Time of Clock */
	bdstime_t toe;	/*!< Time of Ephemeris */
	double sv_cb;   //SV clock bias (seconds)
	double sv_cd;   //SV clock bias (seconds)
	double sv_cdr;  // SV clock drift rate (sec/sec2)
	//BROADCAST ORBIT–1
	int aode;	/*!< Isuse of Data, Ephemeris */
	double crs;
	double delta_n;	/*!< Delta-N (radians/sec) */
	double m0;/*!< Mean anamoly (radians) */
	//BROADCAST ORBIT–2
	double cuc;	/*!< Cuc (radians) */
	double ecc;/*!< e Eccentricity */
	double cus;	/*!< Cus (radians) */
	double sqrt_a;/*!< sqrt(A) (sqrt(m)) */
	//BROADCAST ORBIT–3
	  //Toe Time of Ephemeris
	double cic;	/*!< Correction to inclination cos (radians) */
	double omega0;/*!< Longitude of the ascending node (radians) */
	double cis;	/*!< Correction to inclination sin (radians) */
	//BROADCAST ORBIT–4
	double i0;
	double crc;	/*!< Correction to radius cos (meters) */
	double omega;
	double omega_dot;/*!< Omega dot (radians/s) */
	//BROADCAST ORBIT–5
	double idot;/*!< IDOT (radians/s) */
	  //spare
	double bdt_week;
	//spare
  //BROADCAST ORBIT–6
	double sv_accuracy;	/*!< Inclination (radians) */
	double sath1;
	double tgd1;	/*!< TGD1 B1/B3 (seconds) */
	double tgd2;	/*!<  TGD2 B2/B3 (seconds) */
	//BROADCAST ORBIT–7
	double ttom; // Transmission time of message (sec of BDT week)
	double aodc;	/*!< Issue of Data, Clock */
	  //spare
	  //spare
	//double af0 = sv_cb;
	//double af1 = sv_cd;
	//double af2 = sv_cdr;
	//double sqrta = sqrt_a;
	//double A = sqrt_a * sqrt_a;
	////double sq1e2 = sqrt(1 - ecc * ecc);
	//double inc0 = i0;
	//double omg0 = omega0;
	//double omgkdot = omega_dot;
	//double aop = omega;
	// 
	// 
	//double tgd;	/*!< Group delay L2 bias */
	//int svhlth;
	//int codeL2;
	//// Working variables follow
	//double n; 	/*!< Mean motion (Average angular velocity) */
	//double sq1e2;	/*!< sqrt(1-e^2) */
	//double A;	/*!< Semi-major axis */
	//double omgkdot; /*!< OmegaDot-OmegaEdot */
} ephem_BDS_t;
typedef struct
{
	int enable;
	int vflg;
	double alpha0,alpha1,alpha2,alpha3;
	double beta0,beta1,beta2,beta3;
	double A0,A1;
	int dtls,tot,wnt;
	int dtlsf,dn,wnlsf;
} ionoutc_t;


typedef struct
{
	bdstime_t g;
	double range; // pseudorange
	double rate;
	double d; // geometric distance
	double azel[2];
	double iono_delay;
} range_t;

typedef struct
{
	double f_B1cCode;
	double B1c_CodePhase;
	double B1c_CarrPhase;
	double B1c_CodeStep;
	double B1c_CarrStep;
	int B1c_iword;
	int B1c_ibit;
	int B1c_icode;//icode 二级扩频码计数器

	double fa_phase;
	double fa_step;
	double fb_phase;
	double fb_step;

} B1c_cal;


/*! \brief Structure representing a Channel */
typedef struct
{
	int prn;	/*< PRN Number */
	float amp;
	int ca[CA_SEQ_LEN*20]; /*< C/A Sequence */
	B1c_cal b1c;
	double B1I_carr;
	double B1I_carrstep;
	double f_carr;	/*< Carrier frequency */
	double f_code;	/*< Code frequency */
	double f_SUBcarr;//子载波频率14*1.023e6
	//unsigned int carr_phase; /*< Carrier phase */
	double carr_phase;
	double carr_phasestep;	/*< Carrier phasestep */ //float精度是否满足
	double code_phasestep;
	double code_phase; /*< Code phase */
	double IFcarr_phase;
	double IFcarr_step;
	bdstime_t g0;	/*!< time at start */
	unsigned long sbf[5][N_DWRD_SBF]; /*!< current subframe */
	unsigned long dwrd[N_DWRD]; /*!< Data words of sub-frame */
	int iword;	/*!< initial word */
	int ibit;	/*!< initial bit */
	int icode;	/*!< initial code */
	int dataBit;	/*!< current data bit */
	int codeCA;	/*!< current C/A code */
	double azel[2];
	range_t rho0;
} channel_t;

typedef struct
{
	bdstime_t g0;
	range_t rh0;
	double carrierphase;
	ephem_BDS_t eph[EPHEM_ARRAY_SIZE][MAX_BDS_SAT];
	ionoutc_t ionoutc;
	int neph;
	channel_t chan[MAX_CHAN];
	double(*xyz)[3];
	datetime_t t0, tmin, tmax;
	bdstime_t gmin, gmax, grx;
	int allocatedSat[MAX_BDS_SAT];
	char* navbit;
	char* b1c_nav;
	int NH[20];
}transfer_parameter;

typedef std::complex<float> complexf;
int bds_sim(Table* look_up, transfer_parameter* tp, cuFloatComplex* buff, size_t max_len, int freq_samp, int fc, int tsim, float* dev_i_buff);
unsigned char quantify(float src, int mode);
int read_BDS_RinexNav_All(ephem_BDS_t eph[][MAX_BDS_SAT], ionoutc_t* ionoutc, const char* fname);//读取的是fname阶段
int readUserMotion(double(*xyz)[3], const char* filename, int tsim);
void BDB1Icodegen(int* p, int num);
cudaError_t checkCuda(cudaError_t result);
#endif
