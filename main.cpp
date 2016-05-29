#include <helper_functions.h>

#include <stdio.h>
#include <assert.h>
#include<stdlib.h>

#include "struct.h"
#include "init_cuda.h"
#include "cuda_scft.h"



// CUDA runtime

int main(int argc, char **argv){

	long NxNyNz,ijk;
	GPU_INFO gpu_info;
	CUFFT_INFO cufft_info;

	FILE *dp;
	
	
	cufft_info.Nx=16;	
	cufft_info.Ny=16;
	cufft_info.Nz=16;
	cufft_info.batch=1;
	

	cufft_info.NsA=50;
	
	cufft_info.lx=4.8989794855663562;
	cufft_info.ly=4.8989794855663562;
	cufft_info.lz=4.8989794855663562;
		
	cufft_info.ds0=0.01;
	init_cuda(&gpu_info,0);

	initialize_cufft(&gpu_info,&cufft_info);
		
	//test(&gpu_info,&cufft_info);
	sovDifFft(&gpu_info,&cufft_info,cufft_info.qa_cu,cufft_info.wa_cu,10,1);
	
	finalize_cufft(&gpu_info,&cufft_info);
	//test(&gpu_information);
	

	return 0;
}
