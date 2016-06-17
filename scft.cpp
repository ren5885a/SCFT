#include"scft.h"
#include <assert.h>
#include<stdlib.h>
#include <ctime>

int fftw3_test(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){

	
	int i,j,k;
	double *array;
	int Nx=cufft_info->Nx;
	int Ny=cufft_info->Ny;
	int Nz=cufft_info->Nz;
	array=(double *)malloc(sizeof(double)*cufft_info->NxNyNz);
	
	double *in;
	fftw_complex *in_one;
	fftw_complex *out,*out_one;

	in=(double *)malloc(sizeof(double)*cufft_info->NxNyNz);
	out=(fftw_complex *)malloc(sizeof(fftw_complex)*cufft_info->NxNyNz);

	in_one=(fftw_complex *)malloc(sizeof(fftw_complex)*cufft_info->NxNyNz);
	out_one=(fftw_complex *)malloc(sizeof(fftw_complex)*cufft_info->NxNyNz);

	fftw_plan plan_forward;
	int temp;

	int rank=3;int n[3];
	n[0]=cufft_info->Nx;n[1]=cufft_info->Ny;n[2]=cufft_info->Nz;
	unsigned flags;
	
	for(long ijk=0;ijk<cufft_info->NxNyNz;ijk++) {
		
				
		array[ijk]=ijk;
		
		in[ijk]=ijk;
		
	}
	int Nxh1=Nx/2+1;
	cufft_info->Nxh1NyNz=Nxh1*Ny*Nz;
	long ijk,ijkr;
	/*fftw_plan fftw_plan_dft_2d(int n0, int n1,
                                fftw_complex *in, fftw_complex *out,
                                int sign, unsigned flags);*/
	
	
	for(i=0;i<cufft_info->Nz;i++){
		plan_forward=fftw_plan_dft_r2c_3d(1, Ny, Nx, in+Nx*Ny*i, out+Nx*Ny*i, FFTW_ESTIMATE);
		
		fftw_execute ( plan_forward );
	}
	
	
	for(i=0;i<Nxh1;i++){
		for(j=0;j<cufft_info->Ny;j++){
			
			for(k=0;k<Nz;k++) {in_one[k][0]=out[i+j*Nxh1+k*Nx*Ny][0];in_one[k][1]=out[i+j*Nxh1+k*Nx*Ny][1];}
			//if(i==0&&j==0) for(k=0;k<Nz;k++)  printf("%g %g\n",in_one[k][0],in_one[k][1]);
			plan_forward=fftw_plan_dft_1d(Nz, in_one, out_one,FFTW_FORWARD, FFTW_ESTIMATE);
			fftw_execute ( plan_forward );
			for(k=0;k<Nz;k++) {out[i+j*Nxh1+k*Nx*Ny][0]=out_one[k][0];out[i+j*Nxh1+k*Nx*Ny][1]=out_one[k][1];}
		}
	}
	
	FILE *dp;
	dp=fopen("fftw3_compare.dat","w");

	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++)
	for(i=0;i<Nx;i++){
		ijkr=(long)((k*Ny+j)*Nx+i);
		ijk=(long)(i+Nx*j+Nxh1*Ny*k);
		if(abs(out[ijkr][0])>0.0000001||abs(out[ijkr][1])>0.0000001)
		fprintf(dp,"%d %g %g\n",ijk,out[ijkr][0],out[ijkr][1]);
	}
	fclose(dp);
	
	for(i=0;i<Nxh1;i++){
		for(j=0;j<cufft_info->Ny;j++){
			for(k=0;k<Nz;k++) {in_one[k][0]=out[i+j*Nxh1+k*Nx*Ny][0];in_one[k][1]=out[i+j*Nxh1+k*Nx*Ny][1];}
			// for(k=0;k<Nz;k++)  printf("%g %g\n",in_one[k][0],in_one[k][1]);
			plan_forward=fftw_plan_dft_1d(Nz, in_one, out_one,FFTW_BACKWARD, FFTW_ESTIMATE);
			fftw_execute ( plan_forward );
			//for(k=0;k<Nz;k++)  printf("%g %g\n",out_one[k][0],out_one[k][1]);
			for(k=0;k<Nz;k++) {out[i+j*Nxh1+k*Nx*Ny][0]=out_one[k][0];out[i+j*Nxh1+k*Nx*Ny][1]=out_one[k][1];}
			//for(k=0;k<Nz;k++)  printf("%g %g\n",out_one[k][0],out_one[k][1]);
		}
	}
	for(i=0;i<16;i++) {out[i][0]=i;out[i][1]=0;}
	plan_forward=fftw_plan_dft_c2r_3d(2, 8, 1,  out,in, FFTW_ESTIMATE);
	fftw_execute ( plan_forward );
	/*
	for(i=0;i<cufft_info->Nz;i++){
		//for(k=0;k<8;k++)  printf("%g %g\n",out[k+8*i][0],out[k+8*i][1]);
		plan_forward=fftw_plan_dft_c2r_3d(1, 8, 1,  out+Nx*Ny*i,in+Nx*Ny*i, FFTW_ESTIMATE);
		
		fftw_execute ( plan_forward );
	}
	*/
	//for(i=0;i<16;i++) {printf("%g\n",in[i]);}
	/*	
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++)
	for(i=0;i<Nxh1;i++){
		
		//ijk=(long)((k*Ny+j)*Nxh1+i);
		ijkr=(long)((k*Ny+j)*Nx+i);
		ijk=(long)(k*Nx*Ny+j*Nxh1+i);
		if(abs(out[ijk][0])>=0.00000000001||abs(out[ijk][1])>=0.00000000001){
			
			fprintf(dp,"%d %g %g %g\n",ijkr,out[ijk][0],out[ijk][1],in[ijkr]/512);
		}
		
	}
	*/
	
	printf("finish writiing\n");
	
	free(array);
	free(in);
	free(out);
	return 1;
}
