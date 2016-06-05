#include"struct.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include "init_cuda.h"
#include <errno.h>
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template <class T>
__global__ void
reduce3(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
	
	
	unsigned int tid = threadIdx.x;
	
	T temp;
	T mySum = 0;
	
      for (unsigned int s=0; s<(n/blockDim.x+1); s++) {
	    temp=((s+threadIdx.x*(n/blockDim.x+1))<n)?g_idata[s+threadIdx.x*(n/blockDim.x+1)] : 0;
		
            mySum = mySum + temp;
    

        __syncthreads();
    }
    sdata[tid] = mySum;
	__syncthreads();
  mySum=0;
 	//printf("%d %g\n",tid,sdata[tid]);
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
		 
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }
//	printf("%d %g\n",tid,sdata[tid]);
    // write result for this block to global mem
    if (tid == 0) g_odata[threadIdx.x] = sdata[0];
}

__global__ void initilize_wdz(double *w,double *wdz,double ds2);
__global__ void initilize_q(double *q,double *qInt,int ns1);
__global__ void initilize_in(cufftDoubleComplex *in,double *g,double *wdz,int ns1,int iz);
__global__ void display_GPU_double_data2(double *data,int N);
__global__ void display_GPU_complex_data2(cufftDoubleComplex *data,int N);
__global__ void minus_average(double *data,double average_value);
__global__ void initialize_GPU_cufftDoubleComplex_data(cufftDoubleComplex *data,int index,int NxNyNz);

extern void average_value(std::vector<double*> data,GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
	int gpu_index;
	int threads=1024;
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	std::vector<double*> average_cu;
	double *average;
	double average_value;

	average_cu.resize(gpu_info->GPU_N);
	average=(double *)malloc(sizeof(double)*gpu_info->GPU_N);
	
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		checkCudaErrors(cudaMalloc(&(average_cu[gpu_index]), sizeof(double)));
		
		reduce3<double><<< 1, threads, smemSize >>>(data[gpu_index], average_cu[gpu_index], cufft_info->NxNyNz_gpu);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(&average[gpu_index],  average_cu[gpu_index],sizeof(double),cudaMemcpyDeviceToHost));
		
		cudaDeviceSynchronize();
	}
	
	average_value=0;	
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
		average_value+=average[gpu_index];
		
		
		cudaFree(average_cu[gpu_index]);
		
	}

	average_value=average_value/cufft_info->NxNyNz;
	
	dim3 Grid(cufft_info->Nx_cu,cufft_info->Ny_cu),Block(cufft_info->Nz_cu);
	
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));	
		minus_average<<<Grid,Block>>>(data[gpu_index],average_value);
		
	}
	cudaDeviceSynchronize();
	

	
	free(average);
}

extern void sovDifFft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info,std::vector<double*> g,std::vector<double*> w,int ns,int sign){
	int ns1=ns+1;
	int Nx_cu=cufft_info->Nx_cu;
	int Ny_cu=cufft_info->Ny_cu;
	int Nz_cu=cufft_info->Nz_cu;
	int gpu_index;	
	int iz;
	dim3 grid(1,8,1),block(16,1,1);
	
	
	cufftDoubleComplex *array;
	
	array=(cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*cufft_info->NxNyNz);
	
	for(int i=0;i<cufft_info->NxNyNz;i++){
		array[i].x=i;
		array[i].y=0;
	}
	
	
	
	
	checkCufft(cufftXtMemcpy (cufft_info->plan_forward, cufft_info->device_in, array, CUFFT_COPY_HOST_TO_DEVICE)); 
	checkCufft(cufftXtExecDescriptorZ2Z(cufft_info->plan_forward, cufft_info->device_in, cufft_info->device_in, CUFFT_FORWARD));
	//checkCufft(cufftXtExecDescriptorZ2Z(cufft_info->plan_forward, cufft_info->device_in, cufft_info->device_in, CUFFT_INVERSE));
	checkCufft(cufftXtMemcpy (cufft_info->plan_forward, cufft_info->device_out, cufft_info->device_in, CUFFT_COPY_DEVICE_TO_DEVICE)); 

	
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
				
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		initialize_GPU_cufftDoubleComplex_data<<<grid,block>>>((cufftDoubleComplex*)cufft_info->device_out->descriptor->data[gpu_index],gpu_index,cufft_info->NxNyNz_gpu);
		display_GPU_complex_data2<<<grid,block>>>((cufftDoubleComplex*)cufft_info->device_out->descriptor->data[gpu_index],gpu_info->whichGPUs[gpu_index]);
		cudaDeviceSynchronize();
	
	}
	checkCufft(cufftXtMemcpy (cufft_info->plan_forward, array, cufft_info->device_out, CUFFT_COPY_DEVICE_TO_HOST)); 
	
	FILE *dp;
		
	if((dp=fopen("cufft_compare.dat","w"))==false) printf("did not open file\n");
		for(int i=0;i<cufft_info->NxNyNz;i++){
			if(array[i].x!=0||array[i].y!=0)
			fprintf(dp,"%d %g %g\n",i,array[i].x,array[i].y);
		}
	
	fclose(dp);
	printf("over\n");
	cudaEvent_t start,stop;
	


	float msec;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start,NULL));

	average_value(w,gpu_info,cufft_info);
	
	checkCudaErrors(cudaEventRecord(stop,NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventElapsedTime(&msec,start,stop));
	printf("timea=%g \n",msec);
	checkCudaErrors(cudaEventRecord(start,NULL));

	//checkCufft(cufftXtExecDescriptorZ2Z(cufft_info->plan_forward, cufft_info->device_in, cufft_info->device_in, CUFFT_FORWARD));
	checkCudaErrors(cudaEventRecord(stop,NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventElapsedTime(&msec,start,stop));
	printf("timeb=%g \n",msec);
	
/*
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
		
		
		
		//initilize_wdz<<<grid,block>>>(w[gpu_index],cufft_info->wdz_cu[gpu_index],cufft_info->ds2);
		
		cudaDeviceSynchronize();
		//display_GPU_double_data2<<<10,10>>>(cufft_info->wdz_cu[gpu_index],gpu_index);
		
		
		
	}
	printf("%d %d %d\n",Nx_cu,Ny_cu,Nz_cu);
	
		if(sign==1){
			for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
				checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
				//initilize_q<<<grid,block>>>(g[gpu_index],cufft_info->qInt_cu[gpu_index],ns1);
				cudaDeviceSynchronize();
			}
			for(iz=1;iz<=1;iz++){
				for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
					checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
					
					//initilize_q<<<grid,block>>>(g[gpu_index],cufft_info->qInt_cu[gpu_index],ns1);
					initilize_in<<<grid,block>>>((cufftDoubleComplex*)cufft_info->device_in->descriptor->data[gpu_index],g[gpu_index],cufft_info->wdz_cu[gpu_index],ns1,iz);//(double*)cufft_info->device_in->descriptor->data[gpu_index]
					cudaDeviceSynchronize();
				}
				for(gpu_index=1;gpu_index<4;gpu_index++){	
				
				checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
				//display_GPU_complex_data2<<<16,1>>>((cufftDoubleComplex*)cufft_info->device_in->descriptor->data[gpu_index],gpu_info->whichGPUs[gpu_index]);
				}
				
				printf("here is normal\n");
				//checkCufft(cufftXtExecDescriptorZ2Z(cufft_info->plan_forward, cufft_info->device_in, cufft_info->device_in, CUFFT_FORWARD));
				 
				
				cudaDeviceSynchronize();
				getLastCudaError("Kernel execution failed [  ]");
				
			}
			

		}
		
		else if(sign==-1){

	

		}	
	
	*/
}

__global__ void initilize_wdz(double *w,double *wdz,double ds2){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;
	DIM=blockDim.x*blockDim.y*blockDim.z;
	ij=i+j*DIM;
	wdz[ij]=exp(-w[ij]*ds2);
	__syncthreads();
}

__global__ void initilize_q(double *q,double *qInt,int ns1){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ijk;
	DIM=blockDim.x*blockDim.y*blockDim.z;
	ijk=i+j*DIM;
	q[ijk*ns1]=qInt[ijk];
	
	__syncthreads();
}
__global__ void initilize_in(cufftDoubleComplex *in,double *g,double *wdz,int ns1,int iz){

	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ijk;
	DIM=blockDim.x*blockDim.y*blockDim.z;
	ijk=i+j*DIM;
	in[ijk].x=ijk;//g[ijk*ns1+iz-1]*wdz[ijk];
	in[ijk].y=0;
	__syncthreads();

}
__global__ void display_GPU_double_data2(double *data,int N){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	printf("gpu%d :%ld %g \n",N,ij,data[ij]);
	__syncthreads();
}
__global__ void display_GPU_complex_data2(cufftDoubleComplex *data,int N){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	if(data[ij].x!=0||data[ij].y!=0)
	printf("gpu%d :%ld %g %g\n",N,ij,data[ij].x,data[ij].y);
	__syncthreads();
}
__global__ void minus_average(double *data,double average_value){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	
	data[ij]=data[ij]-average_value;
	

}
__global__ void initialize_GPU_cufftDoubleComplex_data(cufftDoubleComplex *data,int index,int NxNyNz){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	
	data[ij].x=ij+index*NxNyNz;
	data[ij].y=0;
	__syncthreads();
}





