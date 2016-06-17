#include"cuda_aid.cuh"

__global__ void display_GPU_Complex_block_data(cufftDoubleComplex *data,int block_index,int NGPU){
	
	long ij=block_index*(gridDim.x*gridDim.y)+(blockIdx.x+blockIdx.y*gridDim.x)+blockIdx.z*(gridDim.x*gridDim.y*NGPU);
	
	
	printf(" :%ld %g \n",ij,data[ij].x);
	
	__syncthreads();
}

__global__ void display_GPU_Complex_pblock_data(cufftDoubleComplex *data,int GPU_index,int NGPU){
	
	long ij=GPU_index*gridDim.z+blockIdx.z+(blockIdx.x+blockIdx.y*gridDim.x)*(gridDim.z*NGPU);
	
	
	printf(" :%ld %g \n",ij,data[ij].x);
	
	__syncthreads();
}


__global__ void display_GPU_Complex_data(cufftDoubleComplex *data,int index){
	long i=threadIdx.x+threadIdx.y*blockDim.x;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	
	if(abs(data[ij].x)>0.00000001||abs(data[ij].y)>0.00000001){
	printf("gpu%d :%ld %g %g\n",index,ij,data[ij].x,data[ij].y);
	
	}
	__syncthreads();
}

__global__ void display_GPU_double_data(double *data,int N){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;
	long divide;
	DIM=blockDim.x*blockDim.y*blockDim.z;
	divide=gridDim.x*gridDim.y*gridDim.z*4;
	
	ij=i+j*DIM;
	
	printf("gpu%d :%ld %g \n",N,ij,data[ij]/divide);
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

__global__ void initialize_GPU_cufftDoubleComplex_data2(cufftDoubleComplex *data,int index,int NxNyNz){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	
	data[ij].x=ij+NxNyNz*index;
	data[ij].y=0;
	__syncthreads();
}

__global__ void Memerycopy_cufftDoubleComplex(cufftDoubleComplex *data_in,cufftDoubleComplex *data_out){
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	long DIM,ij;

	DIM=blockDim.x*blockDim.y*blockDim.z;
	
	ij=i+j*DIM;
	
	data_out[ij].x=data_in[ij].x;
	data_out[ij].y=data_in[ij].y;
	__syncthreads();
}

//! this programm is written to copy data in from GPU to block

__global__ void copy_data_in_block(cufftDoubleComplex *data_in,cufftDoubleComplex *data_block,int block_index,int block_Num){

	
	
	long ij,ij_cu;
	
	long NxNy=gridDim.x*gridDim.y;
	
	ij_cu=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;
	
	ij=blockIdx.x+blockIdx.y*gridDim.x+NxNy*block_index+blockIdx.z*NxNy*block_Num;
	
	//if(block_index==2) printf("%ld %ld\n",ij,ij_cu);
	data_block[ij_cu]=data_in[ij];
	
	__syncthreads();


}
__global__ void copy_block_back(cufftDoubleComplex *data_out,cufftDoubleComplex *data_block,int block_index,int block_Num){

	long ij,ij_cu;
	
	long NxNy=gridDim.x*gridDim.y;
	
	ij_cu=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;
	
	ij=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*NxNy+(block_index)*gridDim.z*NxNy;
	
	if(block_index==2) printf("%ld %ld\n",ij,ij_cu);
	data_out[ij]=data_block[ij_cu];
	
	__syncthreads();



}
__global__ void matrix_transpose_block(cufftDoubleComplex *data_in_cu,cufftDoubleComplex *data_out_cu,int block_index,int GPU_index,int NGPU){

	long ij_in,ij_out;
	
	
	

	ij_in=block_index*(gridDim.x*gridDim.y)+(blockIdx.x+blockIdx.y*gridDim.x)+blockIdx.z*(gridDim.x*gridDim.y*NGPU);
	
	ij_out=GPU_index*gridDim.z+blockIdx.z+(blockIdx.x+blockIdx.y*gridDim.x)*(gridDim.z*NGPU);
	
	
	data_out_cu[ij_out].x=data_in_cu[ij_in].x;
	data_out_cu[ij_out].y=data_in_cu[ij_in].y;
	
	
	//printf("inverse %ld %ld %g %g %d %d %d\n",ij_in,ij_out,data_in_cu[ij_in].x,data_out_cu[ij_out].x,blockIdx.x,blockIdx.y,blockIdx.z);
__syncthreads();
}
__global__ void matrix_transpose_block_back(cufftDoubleComplex *data_in_cu,cufftDoubleComplex *data_out_cu,int block_index,int GPU_index,int NGPU){

	long ij_in,ij_out;

	ij_in=block_index*(gridDim.x*gridDim.y)+(blockIdx.x+blockIdx.y*gridDim.x)+blockIdx.z*(gridDim.x*gridDim.y*NGPU);
	
	ij_out=GPU_index*gridDim.z+blockIdx.z+(blockIdx.x+blockIdx.y*gridDim.x)*(gridDim.z*NGPU);
	
	data_in_cu[ij_in].x=data_out_cu[ij_out].x;
	data_in_cu[ij_in].y=data_out_cu[ij_out].y;
	
	
	//printf("inverse %ld %ld %g %g %d %d %d\n",ij_in,ij_out,data_in_cu[ij_in].x,data_out_cu[ij_out].x,blockIdx.x,blockIdx.y,blockIdx.z);
__syncthreads();
}













