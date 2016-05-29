#include <cuda_runtime.h>
#include <cufft.h>
#ifndef TAUCS_H
#define TAUCS_H
#include<cufftXt.h>
#include <vector>
//Header stuff here
#define Pi 3.1415926535

typedef struct {
	int Nx;
	int Ny;
	int Nz;
	double *data_real_in;
	double *data_real_out;
	cufftDoubleComplex *data_com_in;
	cufftDoubleComplex *data_com_out;
}data_assem;


typedef struct {
	cudaDeviceProp prop[64];
	int GPU_N;
	int *whichGPU;
	
}GPU_INFO;
typedef struct{
	cufftHandle plan_forward,plan_backward;

	int Nx;
	int Ny;
	int Nz;
	int Nx_cu;
	int Ny_cu;
	int Nz_cu;
	
	long NxNyNz;	//!total grid number
	long NxNyNz_gpu;	//! grid number in each GPU
	int NsA;
	int NsB;
	int ns;	
	
	double lx,ly,lz;
	double dx,dy,dz;
	double ds0,ds2; 
	
	double *kx,*kz,*ky;

	int batch;
	double *in;
	cufftDoubleComplex *out;
	
	cudaLibXtDesc *device_in;
	cudaLibXtDesc *device_out;
	cudaStream_t *stream;
	
	double **kxyzdz_cu;//! pointer which is to kxyzdz in each gpu, the same in each GPU not acoording to grid.	
	double *kxyzdz;//!	pointer to CPU
	
	double *wa;
	double *wb;
	double *wc;
	
	double *pha;
	double *phb;
	double *phc;
	
	std::vector<double*> qa_cu;
	std::vector<double*> wa_cu;
	std::vector<double*> wb_cu;
	std::vector<double*>qInt_cu;
	std::vector<double*> wdz_cu;
	double *qA;
	double *qB;
	double *qC;

}CUFFT_INFO;

#endif //TAUCS_H
