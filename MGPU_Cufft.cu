#include"MGPU_Cufft.h"
#include "cuda_aid.cuh"



extern int cufftMultiGPUPlan(GPU_INFO *gpu_info,int NGPU,int Nx,int Ny,int Nz,MGPU_HANDLE *mgpu_handle){
	
	
	mgpu_handle->rank_xy=2;
	mgpu_handle->rank_z=1;
	
	mgpu_handle->Nx=Nx;
	mgpu_handle->Ny=Ny;
	mgpu_handle->Nz=Nz;
	
	
	
	

	mgpu_handle->plan_z=(cufftHandle*)malloc(sizeof(cufftHandle)*NGPU);
	mgpu_handle->plan_xy=(cufftHandle*)malloc(sizeof(cufftHandle)*NGPU);
	mgpu_handle->plan_xy_back=(cufftHandle*)malloc(sizeof(cufftHandle)*NGPU);
	
	mgpu_handle->dim_xy[0]=Ny;
	mgpu_handle->dim_xy[1]=Nx;
	mgpu_handle->dim_z[0]=Nz;
	
	mgpu_handle->NxNy=Nx*Ny;
	mgpu_handle->Nxh1=Nx/2+1;

	mgpu_handle->Nz_cu=mgpu_handle->Nz/NGPU;
	mgpu_handle->Ny_cu=mgpu_handle->Nz/NGPU;
	
	mgpu_handle->Nxh1Ny=mgpu_handle->Nxh1*mgpu_handle->Ny;
	mgpu_handle->Nxh1Ny_cu=mgpu_handle->Nxh1*Ny/NGPU;


	mgpu_handle->Nx_block=mgpu_handle->Nxh1;
	mgpu_handle->Ny_block=Ny/NGPU;
	mgpu_handle->Nz_block=Nz/NGPU;
	mgpu_handle->trans_block_size=mgpu_handle->Nxh1*Ny*Nz/(NGPU*NGPU);

	//printf("%d %d %d\n",Nx,Ny,Nz);
	//printf("%d %d %d\n",mgpu_handle->trans_block_size,mgpu_handle->Nxh1Ny_cu,mgpu_handle->Nxh1);
	
	for (int i=0; i < gpu_info->GPU_N; i++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		

		checkCufft(cufftPlanMany(&mgpu_handle->plan_xy[i],mgpu_handle->rank_xy,mgpu_handle->dim_xy,NULL,1,0,NULL,1,0,CUFFT_D2Z,mgpu_handle->Nz_cu));
		cudaDeviceSynchronize();
		
		checkCufft(cufftPlanMany(&mgpu_handle->plan_xy_back[i],mgpu_handle->rank_xy,mgpu_handle->dim_xy,NULL,1,0,NULL,1,0,CUFFT_Z2D,mgpu_handle->Nz_cu));
		checkCufft(cufftPlanMany(&mgpu_handle->plan_z[i],mgpu_handle->rank_z,mgpu_handle->dim_z,NULL,1,0,NULL,1,0,CUFFT_Z2Z,mgpu_handle->Nxh1Ny_cu));
		
		cudaDeviceSynchronize();		

	}

	
	printf("all init\n");
	mgpu_handle->Trans_matrix.resize(gpu_info->GPU_N*(gpu_info->GPU_N));
	for (int i=0; i < gpu_info->GPU_N; i++){
		for (int j=0; j < gpu_info->GPU_N; j++){
			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
			checkCudaErrors(cudaMalloc(&(mgpu_handle->Trans_matrix[i*(gpu_info->GPU_N)+j]), sizeof(cufftDoubleComplex)* mgpu_handle->trans_block_size));
		}

	}
	printf("all init malloc\n");
	return 1;
}

extern int Matrix_Transpose(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<cufftDoubleComplex*> data_out_cu,std::vector<cufftDoubleComplex*> data_rotate_cu,int sign){
	//! copy data in transpose matrix	
	int gpu_index,block_index;
	dim3 grid(mgpu_handle->Nxh1,mgpu_handle->Ny_block,mgpu_handle->Nz_block);
	if(sign==1){
		
		
		//printf("%d %d %d %d\n",mgpu_handle->Nx_block,mgpu_handle->Ny_block,mgpu_handle->Nz_block,mgpu_handle->Nxh1Ny);
		for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
	
			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
			for(block_index=0;block_index<gpu_info->GPU_N;block_index++){
				matrix_transpose_block<<<grid,1>>>(data_out_cu[gpu_index],data_rotate_cu[block_index],block_index,gpu_index,gpu_info->GPU_N);
			
			}
			cudaDeviceSynchronize();
		}
		
	}
	else if(sign==2){
		
		for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
	
			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
			//display_GPU_Complex_data<<<4,1>>>(data_rotate_cu[gpu_index],gpu_index);
			for(block_index=0;block_index<gpu_info->GPU_N;block_index++){
				matrix_transpose_block_back<<<grid,1>>>(data_out_cu[gpu_index],data_rotate_cu[block_index],block_index,gpu_index,gpu_info->GPU_N);
			
			}
			
		}
		cudaDeviceSynchronize();

	}	
	return 0;

}
extern int FFTMGPU_Exe_Forward(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info,MGPU_HANDLE *mgpu_handle){

	ExeGPUD2Z(gpu_info,mgpu_handle,cufft_info->device_in_cu,cufft_info->device_out_cu);

	Matrix_Transpose(gpu_info,mgpu_handle,cufft_info->device_out_cu,cufft_info->device_rotate_cu,1);
	
	ExeGPUZ2Z(gpu_info,mgpu_handle,cufft_info->device_rotate_cu,cufft_info->device_rotate_cu,1);
	
	Matrix_Transpose(gpu_info,mgpu_handle,cufft_info->device_out_cu,cufft_info->device_rotate_cu,2);

	return 0;
}
extern int FFTMGPU_Exe_Backward(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info,MGPU_HANDLE *mgpu_handle){
	dim3 grid;
	int gpu_index;
	
	
	
	


	ExeGPUZ2Z(gpu_info,mgpu_handle,cufft_info->device_rotate_cu,cufft_info->device_rotate_cu,2);
	
	cudaEvent_t stop,start;
	cudaError_t error;
	float  msecTotal,msec;
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventRecord(start,NULL);
	Matrix_Transpose(gpu_info,mgpu_handle,cufft_info->device_out_cu,cufft_info->device_rotate_cu,2);

	
	error=cudaDeviceSynchronize();
	if(error!=cudaSuccess) printf("something wrong!before \n");
	error=cudaEventRecord(stop,NULL);	
	cudaEventSynchronize(stop);	
			
	error=cudaEventElapsedTime(&msec,start,stop);
			
	printf("time=%0.10f\n",msec);

	ExeGPUZ2D(gpu_info,mgpu_handle,cufft_info->device_in_cu,cufft_info->device_out_cu);
	

	
	
	//
	

	return 0;
}
extern int Test_fft_mgpu(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info,MGPU_HANDLE *mgpu_handle){

	
	int gpu_index;
	cufftHandle plan;
	dim3 grid(cufft_info->Nx,cufft_info->Ny,cufft_info->Nz),block(1,1,1);
	
	cufftDoubleComplex *data_out;
	cufftDoubleReal *data_in;
	cudaMalloc((void**)&data_in,sizeof(cufftDoubleReal)*cufft_info->NxNyNz);
	cudaMalloc((void**)&data_out,sizeof(cufftDoubleComplex)*cufft_info->Nxh1NyNz);
	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));
	initialize_GPU_double_data<<<grid,1>>>(data_in,0,cufft_info->NxNyNz);
	
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
	
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		initialize_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[gpu_index],gpu_index,cufft_info->NxNyNz_gpu);
		//initialize_GPU_cufftDoubleComplex_data2<<<8,block>>>(cufft_info->device_out_cu[gpu_index],gpu_index,cufft_info->Nxh1NyNz_gpu);
		cudaDeviceSynchronize();
	}
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		cudaStreamSynchronize(cufft_info->stream[gpu_index]);
	}
	
	int n[3]={256,256,256};
	if (cufftPlanMany(&plan, 3, n,
				  NULL, 1, 16,
				  NULL, 1, 0,
				  CUFFT_Z2D,1) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return 0;	
	}
	
	
	
	FFTMGPU_Exe_Forward(gpu_info,cufft_info,mgpu_handle);
	
	FFTMGPU_Exe_Backward(gpu_info,cufft_info,mgpu_handle);
	
	//cufftExecD2Z(plan,data_in,data_out);
	//cufftExecZ2D(plan,data_out,data_in);

	
/*	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[0],0);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[1],1);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[2],2);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[3],3);
/*
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[0],0);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[1],1);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[2],2);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[3],3);
/*
	checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[0]));
	
	if (cufftPlanMany(&plan, 3, n,
				  NULL, 1, 16,
				  NULL, 1, 0,
				  CUFFT_Z2D,1) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return 0;	
	}
	
	if (cufftExecZ2D(plan, data,data_d) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return 0;		
	}
	
if (cudaDeviceSynchronize() != cudaSuccess){
  	fprintf(stderr, "Cuda error: Failed to synchronize\n");
   	return 0;
}	
	display_GPU_double_data<<<grid,1>>>(data_d,0);
	//display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[0],0);
	//ExeGPUD2Z(gpu_info,mgpu_handle,cufft_info->device_in_cu,cufft_info->device_out_cu);
	/*
	ExeGPUZ2D(gpu_info,mgpu_handle,cufft_info->device_in_cu,cufft_info->device_out_cu);
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[gpu_index],gpu_index);
	}

	//FFTMGPU_Exe_Forward(gpu_info,cufft_info,mgpu_handle);

	
	
	//FFTMGPU_Exe_Backward(gpu_info,cufft_info,mgpu_handle);
	getLastCudaError("reduceKernel() execution failed.\n");
	
	grid.x=cufft_info->Nx;grid.y=cufft_info->Ny;grid.z=cufft_info->Nz_cu;
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		cudaStreamSynchronize(cufft_info->stream[gpu_index]);
	}
	
	printf("results----------------\n");
	/*
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[0],0);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[1],1);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[2],2);
	display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[3],3);
	
	/*
	grid.x=cufft_info->Nxh1;grid.y=cufft_info->Ny;grid.z=cufft_info->Nz_cu;
	display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[0],0);
	cudaDeviceSynchronize();
	display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[1],1);
	cudaDeviceSynchronize();
	display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[2],2);
	cudaDeviceSynchronize();
	display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[3],3);
	//getCudaLastError();
	cudaDeviceSynchronize();

	grid.x=5;grid.y=8;grid.z=1;
	//display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[0],0);
	
	/*
	//display_GPU_double_data<<<grid,1>>>(cufft_info->device_in_cu[0],0);
	ExeGPUD2Z(gpu_info,mgpu_handle,cufft_info->device_in_cu,cufft_info->device_out_cu);
	grid.x=5;grid.y=8;grid.z=2;
	//display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[0],0);

	cudaDeviceSynchronize();
	
	Matrix_Transpose(gpu_info,mgpu_handle,cufft_info->device_out_cu,cufft_info->device_rotate_cu,1);
	
	grid.x=cufft_info->Nz;grid.y=1;grid.z=1;
	//display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_rotate_cu[0],0);
	ExeGPUZ2Z(gpu_info,mgpu_handle,cufft_info->device_rotate_cu,cufft_info->device_rotate_cu);
	
	
	grid.x=5;grid.y=8;grid.z=2;
	//display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_rotate_cu[0],0);
	cudaDeviceSynchronize();
	printf("--------");
	Matrix_Transpose(gpu_info,mgpu_handle,cufft_info->device_rotate_cu,cufft_info->device_out_cu,2);
	grid.x=5;grid.y=8;grid.z=1;
	display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[0],0);
	//display_GPU_Complex_pblock_data<<<grid,1>>>(cufft_info->device_rotate_cu[0],0,4);
	/*
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++)	{
	
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		
		
		if(gpu_index==0){

			display_GPU_Complex_data<<<grid,1>>>(cufft_info->device_out_cu[gpu_index],0);//mgpu_handle->Trans_matrix[i]
			cudaDeviceSynchronize();
			//matrix_transpose<<<grid,1>>>(mgpu_handle->Trans_matrix[0]);
			cudaDeviceSynchronize();
			printf("-------------\n");	
			//display_GPU_Complex_data<<<grid,1>>>(mgpu_handle->Trans_matrix[0],0);//mgpu_handle->Trans_matrix[i]
			cudaDeviceSynchronize();
		}
		
	}
	*/
	return 0;

}
extern int ExeGPUD2Z(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<double*> data_in_cu,std::vector<cufftDoubleComplex*> data_out_cu){

	for (int gpu_index=0; gpu_index < gpu_info->GPU_N; gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
			
			checkCufft(cufftExecD2Z(mgpu_handle->plan_xy[gpu_index],data_in_cu[gpu_index],data_out_cu[gpu_index]));
			//checkCudaErrors(cudaMemcpy(data_in_cu[], g0, buf_size, cudaMemcpyDefault));
				
	}
	cudaDeviceSynchronize();
	return 0;
}
extern int ExeGPUZ2D(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<double*> data_in_cu,std::vector<cufftDoubleComplex*> data_out_cu){

	for (int gpu_index=0; gpu_index < gpu_info->GPU_N; gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
			
			checkCufft(cufftExecZ2D(mgpu_handle->plan_xy_back[gpu_index],data_out_cu[gpu_index],data_in_cu[gpu_index]));
			//checkCudaErrors(cudaMemcpy(data_in_cu[], g0, buf_size, cudaMemcpyDefault));
			
	}
	cudaDeviceSynchronize();	
	return 0;
}
extern int ExeGPUZ2Z(GPU_INFO *gpu_info,MGPU_HANDLE *mgpu_handle,std::vector<cufftDoubleComplex*> data_out_cu,std::vector<cufftDoubleComplex*> data_rotate_cu,int sign){

	for (int gpu_index=0; gpu_index < gpu_info->GPU_N; gpu_index++){

			checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
			if(sign==1){
				checkCufft(cufftExecZ2Z(mgpu_handle->plan_z[gpu_index],data_out_cu[gpu_index],data_rotate_cu[gpu_index],CUFFT_FORWARD));
			}
			else if(sign==2){
				checkCufft(cufftExecZ2Z(mgpu_handle->plan_z[gpu_index],data_out_cu[gpu_index],data_rotate_cu[gpu_index],CUFFT_INVERSE));

			}
				
	}
	cudaDeviceSynchronize();
	return 0;
}




















