#include"struct.h"
#include"stdio.h"

__global__ void display_GPU_Complex_block_data(cufftDoubleComplex *data,int block_index,int NGPU);

__global__ void display_GPU_Complex_pblock_data(cufftDoubleComplex *data,int GPU_index,int NGPU);

__global__ void display_GPU_Complex_data(cufftDoubleComplex *data,int index);

__global__ void display_GPU_double_data(double *data,int N);

__global__ void initialize_GPU_double_data(double *data,int index,int NxNyNz);

__global__ void initialize_GPU_cufftDoubleComplex_data2(cufftDoubleComplex *data,int index,int NxNyNz);

__global__ void Memerycopy_cufftDoubleComplex(cufftDoubleComplex *data_in,cufftDoubleComplex *data_out);

__global__ void copy_data_in_block(cufftDoubleComplex *data_in,cufftDoubleComplex *data_block,int block_index,int block_Num);

__global__ void copy_block_back(cufftDoubleComplex *data_out,cufftDoubleComplex *data_block,int block_index,int block_Num);

__global__ void matrix_transpose_block(cufftDoubleComplex *data_in_cu,cufftDoubleComplex *data_out_cu,int block_index,int GPU_index,int NGPU);

__global__ void matrix_transpose_block_back(cufftDoubleComplex *data_in_cu,cufftDoubleComplex *data_out_cu,int block_index,int GPU_index,int NGPU);
