#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_functions.h>
#include"struct.h"
#include <cufft.h>


#include "init_cuda.h"
/* The number of the grid point should be divisible by GPU number*/

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}
// CUDA includes
#include <cuda_runtime.h>

void checkCufft(cufftResult result)
{
	if(result!=CUFFT_SUCCESS){
		printf(" In line  file, cuFFT did not successfully created the FFT plan, the reason is:\n");
		if(result==CUFFT_INVALID_PLAN) printf("The plan parameter is not a valid handle.\n");
		else if(result==CUFFT_ALLOC_FAILED) printf("The allocation of GPU resources for the plan failed.\n");
		else if(result==CUFFT_INVALID_VALUE)printf("One or more invalid parameters were passed to the API.\n");
		else if(result==CUFFT_INTERNAL_ERROR)printf("An internal driver error was detected.\n");
		else if(result==CUFFT_SETUP_FAILED) printf("The cuFFT library failed to initialize.\n");
		else if(result==CUFFT_INVALID_SIZE) printf("Either or both of the nx or ny parameters is not a supported size.\n");
		else if(result==CUFFT_EXEC_FAILED)printf("cuFFt failed to execute the transform on GPU\n");
		exit(0);
	}

}



__global__ void display_GPU_Complex_data(cufftDoubleComplex *data);
__global__ void display_GPU_double_data(double *data,int N);
__global__ void initialize_GPU_double_data(double *data,int index,int NxNyNz);

extern void init_cuda(GPU_INFO *gpu_info,int display){
	
	int gpu_count;
	int i,j;
	cudaDeviceProp prop[64];
	int *gpuid;
	int can_access_peer_0_1;
	
	gpu_count=0;
	gpuid=(int*)malloc(sizeof(int));
	checkCudaErrors(cudaGetDeviceCount(&gpu_info->GPU_N));
	
	if(gpu_info->GPU_N==8)
	gpu_info->GPU_N=4;
	gpu_info->whichGPUs=(int*)malloc(sizeof(int)*(gpu_info->GPU_N));
	
	for(i=0;i<(gpu_info->GPU_N);i++)
		gpu_info->whichGPUs[i]=i;	//!Define on these GPU to calculate 


	for (i=0; i < gpu_info->GPU_N; i++){
        	checkCudaErrors(cudaGetDeviceProperties(&gpu_info->prop[i], i));
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		// Only boards based on Fermi can support P2P
		
            	gpuid[gpu_count++] = gpu_info->whichGPUs[i];
		if(display==1){
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, gpu_info->prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
			printf("maxThreadsDim %d %d %d\n",gpu_info->prop[i].maxThreadsDim[0],gpu_info->prop[i].maxThreadsDim[1],gpu_info->prop[i].maxThreadsDim[2]);
            		printf("maxThreadsPerBlock %d\n",gpu_info->prop[i].maxThreadsPerBlock);
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
			
		}
		
		for(j=0;j<gpu_info->GPU_N;j++){
			if(i!=j){
				checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpu_info->whichGPUs[i], gpu_info->whichGPUs[j]));
    				
				if(can_access_peer_0_1) {

					
					checkCudaErrors(cudaDeviceEnablePeerAccess(gpu_info->whichGPUs[j], 0));
					
					
				}				
				
				
			}
			

		}
	
        }
	
       // on institue cuda cluster,
	//gpu_info->GPU_N=2;
   
}

extern void initialize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){

	
	int *whichGPUs,i,Dim[3],j,k;
	int rank = 3;
	int Nx=cufft_info->Nx;
	int Ny=cufft_info->Ny;
	int Nz=cufft_info->Nz;
	
	cufft_info->Nx_cu=cufft_info->Nx;
	cufft_info->Ny_cu=cufft_info->Ny;
	cufft_info->Nz_cu=cufft_info->Nz/gpu_info->GPU_N;
	
	long NxNyNz=cufft_info->Nx*cufft_info->Ny*cufft_info->Nz,ijk;
	cufft_info->NxNyNz=NxNyNz;
	cufft_info->NxNyNz_gpu=NxNyNz/gpu_info->GPU_N;
	int batch=cufft_info->batch;
	double dx,dy,dz;
	double temp;
	char comment[200];
	double ksq,ds0;
	ds0=cufft_info->ds0;

	cufft_info->ds2=cufft_info->ds0/2;
	cufft_info->dfB=1-cufft_info->fA;

	cufft_info->NsA = ((int)(cufft_info->fA/cufft_info->ds0+1.0e-8));
	cufft_info->dNsB = ((int)(cufft_info->dfB/cufft_info->ds0+1.0e-8));
	
	FILE *dp;
	//!----------- Initialize GPU memery settings. ------------------------------------------------------	
	
	
	int nGPUs = gpu_info->GPU_N;
	whichGPUs=gpu_info->whichGPUs;
	cufft_info->kxyzdz_cu=(double **)malloc(sizeof(double*)*nGPUs);
	
	printf("Wonderful We have successfully initialized GPU setting.\n");

	cufft_info->NxNyNz_gpu=NxNyNz/nGPUs;
	//-----------! Initialize CUFFT settings. ------------------------------------------------------
	
	
	
	cufft_info->worksize=(size_t *)malloc(sizeof(size_t)*(gpu_info->GPU_N));
	
	checkCufft(cufftCreate(&cufft_info->plan_forward));
	checkCufft(cufftCreate(&cufft_info->plan_backward));

	
	
	
	
	checkCufft(cufftXtSetGPUs (cufft_info->plan_forward, nGPUs, whichGPUs));
	checkCufft(cufftXtSetGPUs (cufft_info->plan_backward, nGPUs, whichGPUs));
	
	Dim[0]=Nz;Dim[1]=Ny;Dim[2]=Nx;
	
	checkCufft(cufftMakePlanMany (cufft_info->plan_forward, rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_Z2Z, batch, cufft_info->worksize));
	checkCufft(cufftMakePlanMany (cufft_info->plan_backward, rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_Z2Z, batch, cufft_info->worksize));
	
	checkCufft(cufftXtMalloc (cufft_info->plan_forward, &cufft_info->device_in, CUFFT_XT_FORMAT_INPLACE)); 
	checkCufft(cufftXtMalloc (cufft_info->plan_backward, (cudaLibXtDesc **)&cufft_info->device_out, CUFFT_XT_FORMAT_INPLACE)); 
	
	
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed [  ]");
	printf("Wonderful We have successfully initialized cufft setting.\n");
	//-----------! Initialize malloc and initilize on CPU. ------------------------------------------------------	
	
	cufft_info->in=(double *)malloc(sizeof(double)*NxNyNz);
	cufft_info->out=(cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*NxNyNz);

	cufft_info->wa=(double*)malloc(sizeof(double)*NxNyNz);
	cufft_info->wb=(double*)malloc(sizeof(double)*NxNyNz);
	cufft_info->pha=(double*)malloc(sizeof(double)*NxNyNz);
	cufft_info->phb=(double*)malloc(sizeof(double)*NxNyNz);

	cufft_info->kx=(double *)malloc(sizeof(double)*Nx);
	cufft_info->ky=(double *)malloc(sizeof(double)*Ny);
	cufft_info->kz=(double *)malloc(sizeof(double)*Nz);

	cufft_info->dx=cufft_info->lx/(double)Nx;
	cufft_info->dy=cufft_info->ly/(double)Ny;
	cufft_info->dz=cufft_info->lz/(double)Nz;
	
	dx=cufft_info->dx;
	dy=cufft_info->dy;
	dz=cufft_info->dz;
	
	cufft_info->kxyzdz=(double *)malloc(sizeof(double)*NxNyNz);	

	for(i=0;i<=Nx/2-1;i++)cufft_info->kx[i]=2*Pi*i*1.0/Nx/dx;
	for(i=Nx/2;i<Nx;i++)cufft_info->kx[i]=2*Pi*(i-Nx)*1.0/dx/Nx;
	for(i=0;i<Nx;i++)cufft_info->kx[i]*=cufft_info->kx[i];

	for(i=0;i<=Ny/2-1;i++)cufft_info->ky[i]=2*Pi*i*1.0/Ny/dy;
	for(i=Ny/2;i<Ny;i++)cufft_info->ky[i]=2*Pi*(i-Ny)*1.0/dy/Ny;
	for(i=0;i<Ny;i++)cufft_info->ky[i]*=cufft_info->ky[i];

    	for(i=0;i<=Nz/2-1;i++)cufft_info->kz[i]=2*Pi*i*1.0/Nz/dz;
    	for(i=Nz/2;i<Nz;i++)cufft_info->kz[i]=2*Pi*(i-Nz)*1.0/dz/Nz;
    	for(i=0;i<Nz;i++)cufft_info->kz[i]*=cufft_info->kz[i];	

	
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++)
	for(i=0;i<Nx;i++)
	{
		ijk=(long)((k*Ny+j)*Nx+i);
		ksq=cufft_info->kx[i]+cufft_info->ky[j]+cufft_info->kz[k];
		cufft_info->kxyzdz[ijk]=exp(-ds0*ksq);
	}

	cufft_info->stream=(cudaStream_t*)malloc( sizeof(cudaStream_t)*gpu_info->GPU_N);
	FILE *fp;
	if(cufft_info->intag==1024){
			for(int i=0;i<cufft_info->batch;i++){
				sprintf(comment,"phi_%d.dat",i+1);
				if((fp=fopen(comment,"r"))==false){
					printf("Configration file %s did not exist, please check it in your directory.\n",comment);
				}
			
				fgets(comment,200,fp);
				fgets(comment,200,fp);
				for(long ijk=0;ijk<cufft_info->NxNyNz;ijk++){
					fscanf(fp,"%lg %lg %lg %lg\n",&cufft_info->pha[ijk],&cufft_info->phb[ijk],&cufft_info->wa[ijk],&cufft_info->wb[ijk]);

				}
				fclose(fp);
			}

	}
	
	
	
	cufft_info->wa_cu.resize(gpu_info->GPU_N);
	cufft_info->wb_cu.resize(gpu_info->GPU_N);
	cufft_info->qa_cu.resize(gpu_info->GPU_N);
	cufft_info->wdz_cu.resize(gpu_info->GPU_N);
	cufft_info->qInt_cu.resize(gpu_info->GPU_N);

	
	printf("Wonderful We have successfully initialized CPU setting.\n");
	//-----------! Initialize malloc and initilize on each GPUs. ------------------------------------------------------	

	//for(ijk=0;ijk<20;ijk++){
		//printf("%g %g\n",cufft_info->wa[ijk],cufft_info->wb[ijk]);
	//}	
	
	
	
	for (i=0; i < gpu_info->GPU_N; i++){

		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		
		
		checkCudaErrors(cudaMalloc((void**)&(cufft_info->kxyzdz_cu[i]), sizeof(double)* NxNyNz));
		checkCudaErrors(cudaMemcpy(cufft_info->kxyzdz_cu[i],  cufft_info->kxyzdz,sizeof(double)*NxNyNz,cudaMemcpyHostToDevice));
		dim3 grid(1,1,1),block(4,16,16);
		
		
		
		checkCudaErrors(cudaMalloc(&(cufft_info->wa_cu[i]), sizeof(double)* cufft_info->NxNyNz_gpu));
		checkCudaErrors(cudaMemcpy(cufft_info->wa_cu[i],  cufft_info->wa+cufft_info->NxNyNz_gpu,sizeof(double)*cufft_info->NxNyNz_gpu,cudaMemcpyHostToDevice));
		//initialize_GPU_double_data<<<grid,block>>>(cufft_info->wa_cu[i],i,cufft_info->NxNyNz_gpu);
		checkCudaErrors(cudaMalloc(&(cufft_info->qInt_cu[i]), sizeof(double)* cufft_info->NxNyNz_gpu));
	
		checkCudaErrors(cudaMalloc(&(cufft_info->wdz_cu[i]), sizeof(double)* cufft_info->NxNyNz_gpu));
		
		checkCudaErrors(cudaMalloc(&(cufft_info->wb_cu[i]), sizeof(double)* cufft_info->NxNyNz_gpu));
		checkCudaErrors(cudaMalloc(&(cufft_info->qa_cu[i]), sizeof(double)* cufft_info->NxNyNz_gpu*(cufft_info->NsA+1)*batch));//cufft_info->NsA
		
		
		
		
	}
	cudaDeviceSynchronize();
	
	
	
	printf("Wonderful We have successfully initialized all the data.\n");
	
	
}




extern void finalize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
	int i,j;
	int can_access_peer_0_1;
	
	//! free memery on GPU
	cudaSetDevice(gpu_info->whichGPUs[0]);
	checkCufft(cufftDestroy(cufft_info->plan_forward));
	checkCufft(cufftDestroy(cufft_info->plan_backward));
	checkCufft(cufftXtFree(cufft_info->device_in));
	checkCufft(cufftXtFree(cufft_info->device_out));
	
	for (i=0; i < gpu_info->GPU_N; i++){
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		checkCudaErrors(cudaFree(cufft_info->kxyzdz_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->qa_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->wa_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->wb_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->wdz_cu[i]));
		for(j=0;j<gpu_info->GPU_N;j++){
			if(i!=j){
				checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
				checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer_0_1, i, j));
    				

				if(can_access_peer_0_1) {

					
					checkCudaErrors(cudaDeviceDisablePeerAccess(gpu_info->whichGPUs[j]));
			
					
				}// end if can_access_peer_0_1			
				
				
			}// end i!=j
			
		
	}// end loop j
	}
	cudaDeviceSynchronize();
	//! free memery on CPU
	
	free(cufft_info->in);
	free(cufft_info->out);
	
	free(cufft_info->wa);
	free(cufft_info->wb);
	free(cufft_info->pha);
	free(cufft_info->phb);
	
	free(cufft_info->kx);
	free(cufft_info->ky);
	free(cufft_info->kz);
	free(cufft_info->stream);
	free(cufft_info->kxyzdz);
	free(gpu_info->whichGPUs);
	free(cufft_info->worksize);
	printf("Wonderful We have successfully evaculate all the memery on GPU and CPU \n");
	
	cudaDeviceReset();
}





extern void test(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
	double *data;
	int i,index;
	dim3 grid(3,3,1),block(4,1,1);
	cudaSetDevice(0);
		
	//cudaStreamCreate(&stream0[d]);	
	data=(double*)malloc(sizeof(double)*(cufft_info->NxNyNz));
	for(i=0;i<cufft_info->NxNyNz;i++){
		data[i]=i;	
	}
	//printf("%d %d\n",cufft_info->NxNyNz_gpu,cufft_info->NxNyNz);
	for(index=0;index<gpu_info->GPU_N;index++){///gpu_info->GPU_N
		checkCudaErrors(cudaSetDevice(index));
		
		checkCudaErrors(cudaMemcpy(cufft_info->wa_cu[index],data+index*cufft_info->NxNyNz_gpu,sizeof(double)*cufft_info->NxNyNz_gpu,cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpyPeerAsync(cufft_info->wa_cu[index],index,cufft_info->wa_cu[0],0,sizeof(double)*cufft_info->NxNyNz_gpu));
		//checkCudaErrors(cudaMemcpy(cufft_info->qa_cu[index],data,sizeof(double)*cufft_info->NxNyNz_gpu,cudaMemcpyHostToDevice));
		//cudaMemcpy(cufft_info->wa_cu[index],data,sizeof(double)*cufft_info->NxNyNz_gpu,cudaMemcpyHostToDevice);
		
	}
	cudaDeviceSynchronize();
	for(index=0;index<gpu_info->GPU_N;index++){
		checkCudaErrors(cudaSetDevice(index));
		//display_GPU_double_data<<<1,block>>>(cufft_info->wa_cu[index],index);
	}
	cufftXtExecDescriptorZ2Z(cufft_info->plan_forward, cufft_info->device_in, cufft_info->device_out, CUFFT_FORWARD);
	//checkCudaErrors(cudaMemcpy(data_cu,data,sizeof(cufftDoubleComplex)*gpu_info->prop[0].maxThreadsPerBlock,cudaMemcpyHostToDevice));	
	//display_GPU_Complex_data<<<grid,block>>>(data_cu);
	
}





extern void com_to_com1d(GPU_INFO *gpu_info,data_assem *data_test){
	cufftHandle plan;
	cufftComplex *data_in,*data_out;
	int BATCH=1;
	
	cudaMalloc((void**)&data_in,sizeof(cufftComplex)*data_test->Nx);
	checkCudaErrors(cudaMalloc((void**)&data_out,sizeof(cufftComplex)*data_test->Nx));
	
	cudaMemcpy(data_in,data_test->data_com_in,sizeof(cufftComplex)*data_test->Nx,cudaMemcpyHostToDevice);

	checkCufft(cufftPlan1d(&plan,data_test->Nx,CUFFT_C2C,BATCH));
	
	cufftExecC2C(plan,data_in,data_out,CUFFT_FORWARD);

	cudaMemcpy(data_test->data_com_out,data_out,sizeof(cufftComplex)*data_test->Nx,cudaMemcpyDeviceToHost);
	printf("dd %g %g\n",data_test->data_com_out[0].x,data_test->data_com_out[0].y);
	cudaFree(data_in);
	cudaFree(data_out);
	cufftDestroy(plan);
	
}
extern void D1_MultipleGPU(GPU_INFO *gpu_info,data_assem *data_test,int N){
	
	cufftHandle plan_input; cufftResult result;
    	result = cufftCreate(&plan_input);
	int nGPUs = 4, whichGPUs[4];
	whichGPUs[0] = 0; whichGPUs[1] = 1;whichGPUs[2] = 2;whichGPUs[3] = 3;
	
	dim3 gridDim(1,1),blockDim(10,10);
	printf("grid size on x=%d y=%d z=%d\n",gridDim.x,gridDim.y,gridDim.z);
	printf("block size on x=%d y=%d z=%d\n",blockDim.x,blockDim.y,blockDim.z);
	
	result = cufftXtSetGPUs (plan_input, nGPUs, whichGPUs);
	if(result!=CUFFT_SUCCESS){
		printf("failed to set GPU\n");
	}
	
	size_t worksize[4]; cufftComplex *host_data_input, *host_data_output; 
	int nx = 1024,ny=8,nz=8, batch = 1, rank = 3, n[3]; 
	n[0] = nx; 
	n[1]=ny;
	n[2]=nz;
	int size_of_data = sizeof(cufftComplex) * nx *ny*nz* batch;
	host_data_input = (cufftComplex*)malloc(size_of_data); 
	host_data_output = (cufftComplex*)malloc(size_of_data);
	printf("length is %d\n",nx);
	//initialize_1d_data (nx, batch, rank, n, inembed, &istride, &idist, onembed, &ostride, &odist, host_data_input, host_data_output);
		
	for(int i=0;i<nx*ny*nz;i++){
		host_data_input[i].x=i;
		host_data_input[i].y=0;
	}
	printf("finish initial\n");
	
	
	checkCufft( cufftMakePlanMany (plan_input, rank, n, NULL, 1, nx, NULL, 1, nx, CUFFT_C2C, batch, worksize));
	//result=cufftMakePlan1d(plan_input, nx, CUFFT_C2C, batch, worksize);
	
	
// cufftXtMalloc() - Malloc data on multiple GPUs 
	cudaLibXtDesc *device_data_input, *device_data_output;
	result = cufftXtMalloc (plan_input, &device_data_input, CUFFT_XT_FORMAT_INPLACE); 
	if(result!=CUFFT_SUCCESS){
		printf("failed 1\n");
	}
	
	result = cufftXtMalloc (plan_input, &device_data_output, CUFFT_XT_FORMAT_INPLACE); 

	
	printf("%zu %zu \n", device_data_input->descriptor->size[0],device_data_input->descriptor->size[1]); 		
	printf("%zu %zu \n", worksize[0],worksize[1]); 
	cudaSetDevice(0);
	display_GPU_Complex_data<<<1,10>>>((cufftDoubleComplex*)device_data_input->descriptor->data[0]);

	if(result!=CUFFT_SUCCESS){
		printf("failed 2\n");
	}
	
	// // cufftXtMemcpy() - Copy data from host to multiple GPUs 
	result = cufftXtMemcpy (plan_input, device_data_input, host_data_input, CUFFT_COPY_HOST_TO_DEVICE); 
	
	// // cufftXtExecDescriptorC2C() - Execute FFT on multiple GPUs 
	
	//cudaSetDevice(0);
	result = cufftXtExecDescriptorC2C (plan_input, device_data_input, device_data_input, CUFFT_FORWARD); 
	
	printf("finish memcpy \n");
	
	// // cufftXtMemcpy() - Copy the data to natural order on GPUs 
	
	result = cufftXtMemcpy (plan_input, device_data_output, device_data_input, CUFFT_COPY_DEVICE_TO_DEVICE); 

	cudaSetDevice(0);
	//display_GPU_Complex_data<<<gridDim,blockDim>>>((cufftComplex*)device_data_output->descriptor->data[0],N);
	cudaDeviceSynchronize();
	if(result!=CUFFT_SUCCESS){
		printf("failed copy data from device to device\n");
	}
	printf("problem 1\n");
// // cufftXtMemcpy() - Copy natural order data from multiple GPUs to host 
	result = cufftXtMemcpy (plan_input, host_data_output, device_data_input, CUFFT_COPY_DEVICE_TO_HOST); 

	for(int i=0;i<8;i++){
		printf("%g %g\n",host_data_output[i].x,host_data_output[i].y);
	}

	// // Print output and check results int output_return = output_1d_results (nx, batch, host_data_input, host_data_output); // 
	// cufftXtFree() - Free GPU memory 

	
	result = cufftXtFree(device_data_input); 
	result = cufftXtFree(device_data_output); 
	// // cufftDestroy() - Destroy FFT plan 
	result = cufftDestroy(plan_input); 
	free(host_data_input); 
	free(host_data_output);

	



	

}


__global__ void display_GPU_Complex_data(cufftDoubleComplex *data){
	long i=threadIdx.x+threadIdx.y*blockDim.x;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	printf("%ld %g %g\n",ij,data[ij].x,data[ij].y);
	__syncthreads();
}

__global__ void display_GPU_double_data(double *data,int N){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	printf("gpu%d :%ld %g \n",N,ij,data[ij]);
	__syncthreads();
}

__global__ void initialize_GPU_double_data(double *data,int index,int NxNyNz){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	
	data[ij]=ij+index*NxNyNz;
	
	__syncthreads();
}


/*
FILE *dp;
	double *kxkykz,testD;
	kxkykz=(double *)malloc(sizeof(double)*cufft_info->NxNyNz);
dp=fopen("kxyzdz.dat","r");
	testD=0;
	for(ijk=0;ijk<cufft_info->NxNyNz;ijk++){
		fscanf(dp,"%lg\n",&kxkykz[ijk]);
		testD+=(kxkykz[ijk]-cufft_info->kxyzdz[ijk])*(kxkykz[ijk]-cufft_info->kxyzdz[ijk]);
	}
	printf("compare %g\n",testD);
*/













