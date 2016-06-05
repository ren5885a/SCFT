#include"scft.h"
#include <assert.h>
#include<stdlib.h>
#include <ctime>

int fftw3_test(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){

	

	double *array;
	array=(double *)malloc(sizeof(double)*cufft_info->NxNyNz);
	
	fftw_complex *in;
	fftw_complex *out;
	in=(fftw_complex *)malloc(sizeof(fftw_complex)*cufft_info->NxNyNz);
	out=(fftw_complex *)malloc(sizeof(fftw_complex)*cufft_info->NxNyNz);
	fftw_plan plan_forward;
	int rank=3;int n[3];
	n[0]=cufft_info->Nz;n[1]=cufft_info->Ny;n[2]=cufft_info->Nx;
	unsigned flags;
	
	for(long ijk=0;ijk<cufft_info->NxNyNz;ijk++) {

		array[ijk]=ijk;
		in[ijk][0]=ijk;
		in[ijk][1]=0;
	}
	plan_forward=fftw_plan_dft_3d(n[0], n[1], n[2], in, out,FFTW_FORWARD, FFTW_ESTIMATE);
	clock_t begin = clock();
	
	fftw_execute ( plan_forward );
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("time is %g\n",elapsed_secs*1000);
	
	FILE *dp;
	dp=fopen("fftw3_compare.dat","w");
	for(int i=0;i<cufft_info->NxNyNz;i++){
		if(out[i][0]!=0||out[i][0]!=0)
		fprintf(dp,"%d %g %g\n",i,out[i][0],out[i][1]);
		
	}


	free(array);
	free(in);
	free(out);
	return 1;
}
