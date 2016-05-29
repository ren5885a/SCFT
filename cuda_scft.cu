#include"struct.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>


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
	
	unsigned long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
    unsigned int tid = threadIdx.x;
    unsigned int ij = j*(gridDim.x*gridDim.y*gridDim.z)+  threadIdx.x;

    T mySum = (ij < n) ? g_idata[ij] : 0;

    if (ij + blockDim.x < n)
        mySum += g_idata[ij+blockDim.x];
	__syncthreads();
    sdata[tid] = mySum;
    if(j==0)
	printf("%d %d %g\n",threadIdx.x,ij,mySum);
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__global__ void initilize_wdz(double *w,double *wdz,double ds2);
__global__ void initilize_q(double *q,double *qInt,int ns1);
__global__ void initilize_in(double *in,double *g,double *wdz,int ns1,int iz);
__global__ void display_GPU_double_data2(double *data,int N);
__global__ void average_GPU_double_data(double *data,double *average);



extern void sovDifFft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info,std::vector<double*> g,std::vector<double*> w,int ns,int sign){
	int ns1=ns+1;
	int Nx_cu=cufft_info->Nx_cu;
	int Ny_cu=cufft_info->Ny_cu;
	int Nz_cu=cufft_info->Nz_cu;
	int gpu_index;	
	int iz;
	dim3 grid(Nx_cu,Ny_cu,1),block(Nz_cu,1,1);
	double average;	
	std::vector<double*> average_cu;
	average_cu.resize(gpu_info->GPU_N);
	int threads=8;
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	for(gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
		checkCudaErrors(cudaSetDevice(gpu_index));
		
		checkCudaErrors(cudaMalloc(&(average_cu[gpu_index]), sizeof(double)));
		
		initilize_wdz<<<grid,block>>>(w[gpu_index],cufft_info->wdz_cu[gpu_index],cufft_info->ds2);
		cudaDeviceSynchronize();
		//display_GPU_double_data2<<<10,10>>>(cufft_info->wdz_cu[gpu_index],gpu_index);
		average_GPU_double_data<<<10,10>>>(cufft_info->wa_cu[gpu_index],average_cu[gpu_index]);
		if(gpu_index==0){
		reduce3<double><<< 16, threads, smemSize >>>(cufft_info->wa_cu[gpu_index], average_cu[gpu_index], 100);
		//display_GPU_double_data2<<<1,100>>>(cufft_info->wa_cu[gpu_index],gpu_index);
		display_GPU_double_data2<<<1,1>>>(average_cu[gpu_index],gpu_index);
		}
		if(sign==1){
			/*
			initilize_q<<<grid,block>>>(g[gpu_index],cufft_info->qInt_cu[gpu_index],ns1);

			for(iz=1;iz<=ns;iz++){
				
				initilize_in<<<grid,block>>>((double*)cufft_info->device_in->descriptor->data[gpu_index],g,cufft_info->wdz_cu[gpu_index],ns1,iz);
			}
			*/

		}
		
		else if(sign==-1){

	

		}	
	}

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
__global__ void initilize_in(double *in,double *g,double *wdz,int ns1,int iz){

	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ijk;
	DIM=blockDim.x*blockDim.y*blockDim.z;
	ijk=i+j*DIM;
	in[ijk]=g[ijk*ns1+iz-1]*wdz[ijk];

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

__global__ void average_GPU_double_data(double *data,double *average){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	average[0]+=data[ij];
	__syncthreads();

}





