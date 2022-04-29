/*
 * CUDA convolutional neural net
 */

#include <iostream>
#include <math.h>
#include "ee155_utils.hxx"
#include "matrix.hxx"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
using namespace std;

const int BS=32;		// The blocks are BS x BS.
const int FILT_SIZE_MAX = 12;	// The filter size (needs not be a power of 2)

///////////////////////////////
// This is the CUDA kernel function for you to write.
//////////////////////////////
//N_inp = length of the input matrix, N_f = length of the filter matrix, bN = how many blocks do we have
__global__ void CNN (float *d_inp, float *d_f, float *d_out, int N_inp, int N_f, int bN)
{
    int rB = blockIdx.x;
    int cB = blockIdx.y;
    int rI = threadIdx.y;
    int cI = threadIdx.x;
    __shared__ float S_in[BS][BS], S_f[FILT_SIZE_MAX][FILT_SIZE_MAX], S_out[BS][BS];
    //printf("In thread with r=(%d,%d) c=(%d,%d)\n", rB,rI,cB,cI);
    //...
    int BS_actual = BS - N_f + 1;
    S_out = 0;
    //loading S_in
    int index_inp = (rB * BS_actual + rI) * N_inp + cB * BS_actual + cI;
    S_in[rI][cI] = d_inp[index_inp];
    //loading S_f
    if ((cI < N_f) & (rI < N_f))
    {
        int index_f = rI * N_f + cI;
        S_f[rI][cI] = d_f[index_f];
    }
    __syncthreads();
    //calculating the result
    if ((cI + N_f - 1) < BS)
    {
        if ((rI + N_f - 1) < BS)
        {
            for (int rF = 0; rF < N_f; rF++)
            {
                for(int cF = 0; cF < N_f; cF++)
                {
                    S_out[rI][cI] += S_in[rF][cF] * S_f[rF][cF];
                }
            }
            __syncthreads();
            int index_out = (rB * BS_actual + rI) * (N_inp - N_f + 1) + cB * BS_actual + cI;
            d_out[index_out] = S_out[cI][rI];
        }
    }
}


///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
//
void Matrix::CNN2 (const Matrix &inp, const Matrix &f, int dummy) {
    auto start = start_time();

    // Allocate input matrix in device memory. It's a nice 2^N size, so don't
    // bother with cudaMallocPitch().
    assert (1<<inp._log2NColsAlc == inp._nCols);
    int numElem = inp.data.size(), sizeBytes = numElem*4;
    int len = sqrt(numElem);
    float *d_inp = NULL;
    cudaError_t err = cudaMalloc((void **)&d_inp, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix 'inp'");

    // Copy inp from host memory to device memory.
    err = cudaMemcpy (d_inp, inp.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix inp from host to device");

    // Allocate device memory for filter. Again, don't bother with
    // cudaMallocPitch(); the filter is small, and Matrix has already picked 
    // a power of 2 columns
    float *d_f = NULL;
    int sizeBytes_f = static_cast<int> (f.data.size()) * 4;
    int len_f = sqrt(f.data.size());
    err = cudaMalloc((void **)&d_f, sizeBytes_f);
    ERR_CHK (err, "Failed to allocate device matrix for the filter f");

    // Copy f from host memory to device memory.
    err = cudaMemcpy (d_f, f.data.data(), sizeBytes_f, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix f from host to device");

    // Allocate device memory for the output matrix. In fact, allocate the
    // entire thing (with padding).
    float *d_out = NULL;
    size_t pitch;
    int len_out = len - len_f + 1;
    //cudaMallocPitch ( void ** devPtr, size_t *pitch, size_t width, size_t height)
    err = cudaMallocPitch((void **)&d_out, &pitch, len_out * 4, len_out);
    ERR_CHK (err, "Failed to allocate device matrix 'out'");
    long int time1 = delta_usec (start);

    // Launch the CUDA Kernel
    start = start_time();
    int bN = ceil (len / (BS - len_f + 1));
    //int bN = bN_temp + 1;
    dim3 thBlocks (bN, bN), threads (BS, BS);

    //switch case for different filter size
    switch (len_f)
    {
    case 1:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 1, bN);
        break;
    case 2:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 2, bN);
        break;
    case 3:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 3, bN);
        break;
    case 4:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 4, bN);
        break;
    case 5:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 5, bN);
        break;
    case 6:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 6, bN);
        break;
    case 7:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 7, bN);
        break;
    case 8:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 8, bN);
        break;
    case 9:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 9, bN);
        break;
    case 10:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 10, bN);
        break;
    case 11:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 11, bN);
        break;
    case 12:
        CNN <<<thBlocks, threads>>> (d_inp, d_f, d_out, len, 12, bN);
        break;
    
    default:
        break;
    }

    err = cudaGetLastError();
    ERR_CHK (err, "Failed to launch or finish CNN_kernel");
    long int time2 = delta_usec (start);

    // Copy the result from device memory to host memory.
    start = start_time();
    // cudaMemcpy2D (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
    err = cudaMemcpy2D (this->data.data(), len_out * 4, d_out, pitch, len_out * 4, len_out, cudaMemcpyDeviceToHost);
    ERR_CHK (err, "Failed to copy result from device to host");

    err = cudaFree(d_inp);
    ERR_CHK (err, "Failed to free CUDA matrix inp");
    err = cudaFree(d_f);
    ERR_CHK (err, "Failed to free CUDA matrix f");
    err = cudaFree(d_out);
    ERR_CHK (err, "Failed to free CUDA matrix out");

    long int time3 = delta_usec (start);
    LOG ("\tCUDA CNN took "<<(time1+time2+time3)/1000000.0<<"sec; "<<(time1/1000000.0)<<"s copy to, "
	<< (time2/1000000.0)<<"s for computation, "<< (time3/1000000.0)<<"s copy back");
}
