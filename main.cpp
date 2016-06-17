#include <helper_functions.h>

#include <stdio.h>
#include <assert.h>
#include<stdlib.h>

#include "struct.h"
#include "init_cuda.h"
#include "cuda_scft.h"
#include "/usr/include/fftw3.h"
#include "init.h"
#include "scft.h"
#include "MGPU_Cufft.h"


#include <ctime>
// z dimension must be larger than Nz/GPU_N>=8
// CUDA runtime

int main(int argc, char **argv){

	long NxNyNz,ijk;
	
	GPU_INFO gpu_info;

	data_assem data_test;
	CUFFT_INFO cufft_info;
	MGPU_HANDLE mgpu_handle;
	init_scft(&cufft_info,&gpu_info,argc, argv);


	


	if(gpu_info.kernal_type==1){

		//sovDifFft(&gpu_info,&cufft_info,cufft_info.qa_cu,cufft_info.wa_cu,10,1);
		//fft_test(&gpu_info,&cufft_info);
		printf("---\n");
		//fftw3_test(&gpu_info,&cufft_info);

		cufftMultiGPUPlan(&gpu_info,gpu_info.GPU_N,cufft_info.Nx,cufft_info.Ny,cufft_info.Nz,&mgpu_handle);

		Test_fft_mgpu(&gpu_info,&cufft_info,&mgpu_handle);
		fftw3_test(&gpu_info,&cufft_info);
		finalize_cufft(&gpu_info,&cufft_info);
	}
	/*
	
	*/
	return 0;
}
