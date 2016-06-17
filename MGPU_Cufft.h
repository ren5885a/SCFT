#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include"struct.h"
#include "init_cuda.h"
typedef struct {
	cufftHandle *plan_z;
	cufftHandle *plan_xy;
	cufftHandle *plan_xy_back;
	
	int rank_xy;
	int rank_z;	
	
	int dim_xy[2];
	int dim_z[1];	
	
	long NxNy;
	long Nxh1Ny;

	int Nx,Ny,Nz;
	int Nxh1;

	int Nz_cu;
	int Ny_cu;
	long Nxh1Ny_cu;
	
	int Nx_block;
	int Ny_block;
	int Nz_block;
	long trans_block_size; 

	std::vector<cufftDoubleComplex*> Trans_matrix;

	std::vector<cufftDoubleComplex*> Rev_matrix;
	
}MGPU_HANDLE;


extern int cufftMultiGPUPlan(GPU_INFO *gpu_info,int NGPU,int Nx,int Ny,int Nz,MGPU_HANDLE *mgpu_handle);

extern int ExeGPUD2Z(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<double*> data_in_cu,std::vector<cufftDoubleComplex*> data_out_cu);

extern int ExeGPUZ2D(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<double*> data_in_cu,std::vector<cufftDoubleComplex*> data_out_cu);

extern int ExeGPUZ2Z(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<cufftDoubleComplex*> data_out_cu,std::vector<cufftDoubleComplex*> data_rotate_cu,int sign);

extern int Matrix_Transpose(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<cufftDoubleComplex*> data_in_cu,int sign);

extern int Test_fft_mgpu(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info,MGPU_HANDLE *mgpu_handle);

