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
	int kernal_type;
	cudaDeviceProp prop[64];
	int GPU_N;
	int *whichGPUs;
	
}GPU_INFO;
typedef struct{
	

	//! scft infomation
	
	int batch;// Number of scft calculation in one GPU.
	int intag;
	double hAB; 
	double fA;
	double fB;
	int MaxIT;//Maximum iteration steps
	
	//! Grid size of the system 
	int Nx;
	int Ny;
	int Nz;
	
	int Nx_cu;
	int Ny_cu;
	int Nz_cu;
	
	long NxNyNz;	//!total grid number
	long Nxh1NyNz;
	
	long Nxh1NyNz_gpu;
	long NxNyNz_gpu;	//! grid number in each GPU

	int Nxh1;

	
	int NsA;
	int NsB;
	int ns;
	int dNsB;
	int dfB;
	
	double lx,ly,lz;
	double dx,dy,dz;
	double ds0,ds2; 	

	// cufft configuration variable

	
	cudaStream_t *stream;
	cufftHandle plan_forward,plan_backward;

	

	//! Temperary variables for cufft

	double *in;
	cufftDoubleComplex *out;

	cufftDoubleComplex *wijk_cu;
	
	cudaLibXtDesc *device_in;
	cudaLibXtDesc *device_out;
	
	

	size_t *worksize;
	
	double *kx,*kz,*ky;	
	
	double *wa;
	double *wb;
	double *wc;
	
	double *pha;
	double *phb;
	double *phc;

	double *cufft_sequence;

	double **kxyzdz_cu;//! pointer which is to kxyzdz in each gpu, the same in each GPU not acoording to grid.	
	double *kxyzdz;//!	pointer to CPU

	std::vector<double*> device_in_cu;
	std::vector<cufftDoubleComplex*> device_out_cu;
	std::vector<cufftDoubleComplex*> device_rotate_cu;

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
