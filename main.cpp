#include <helper_functions.h>

#include <stdio.h>
#include <assert.h>
#include<stdlib.h>

#include "struct.h"
#include "init_cuda.h"
#include "cuda_scft.h"
#include "/usr/include/fftw3.h"
#include "init.h"
#include <ctime>
// z dimension must be larger than Nz/GPU_N>=8
// CUDA runtime

int main(int argc, char **argv){

	long NxNyNz,ijk;
	GPU_INFO gpu_info;
	data_assem data_test;
	CUFFT_INFO cufft_info;

	init_scft(&cufft_info,&gpu_info,argc, argv);
	/*
	FILE *dp;
	
	
	cufft_info.Nx=1;	
	cufft_info.Ny=1;
	cufft_info.Nz=64;
	cufft_info.batch=1;
	

	cufft_info.NsA=50;
	
	cufft_info.lx=4.8989794855663562;
	cufft_info.ly=4.8989794855663562;
	cufft_info.lz=4.8989794855663562;
		
	cufft_info.ds0=0.01;
	init_cuda(&gpu_info,0);

	initialize_cufft(&gpu_info,&cufft_info);
	//D1_MultipleGPU(&gpu_info,&data_test,1024);
	//test(&gpu_info,&cufft_info);
	sovDifFft(&gpu_info,&cufft_info,cufft_info.qa_cu,cufft_info.wa_cu,10,1);
	
	finalize_cufft(&gpu_info,&cufft_info);
	//test(&gpu_information);
	double *array;
	array=(double *)malloc(sizeof(double)*256*256*256);
	fftw_complex *out;
	out=(fftw_complex *)malloc(sizeof(fftw_complex)*256*256*256);
	fftw_plan plan_forward;
	int rank=3;int n[3];
	n[0]=256;n[1]=256;n[2]=256;
	unsigned flags;
	
	for(long ijk=0;ijk<256*256*256;ijk++) array[ijk]=ijk;
	 plan_forward=fftw_plan_dft_r2c_3d(n[0], n[1], n[2], array, out, FFTW_ESTIMATE);
	clock_t begin = clock();
	
	fftw_execute ( plan_forward );
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("time is %g\n",elapsed_secs*1000);
	*/
	return 0;
}
