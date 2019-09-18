//This code has been derived from the sample code Advanced/reduction
//released with NVIDIA CUDA version 5.5

/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __REDUCTION_MAX_H__
#define __REDUCTION_MAX_H__

template <class T>
void reduce_max(int size, int threads, int blocks,
            int whichKernel, T *d_idata, T *d_odata, cudaStream_t stream);


/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_MAX_H_
#define _REDUCE_KERNEL_MAX_H_

#include <stdio.h>
#include "reduction.h"



/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void
reduce0_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] = max(sdata[tid],sdata[tid + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void
reduce1_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] = max(sdata[index],sdata[index + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void
reduce2_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid],sdata[tid + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void
reduce3_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T myMax = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        myMax = max(g_idata[i+blockDim.x],myMax);

    sdata[tid] = myMax;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version unrolls the last warp to avoid synchronization where it
    isn't needed.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce4_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T myMax = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        myMax = max(myMax,g_idata[i+blockSize]);

    sdata[tid] = myMax;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + s]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = myMax = max(myMax, smem[tid + 32]);
        }

        if (blockSize >=  32)
        {
            smem[tid] = myMax = max(myMax, smem[tid + 16]);
        }

        if (blockSize >=  16)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  8]);
        }

        if (blockSize >=   8)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  4]);
        }

        if (blockSize >=   4)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  2]);
        }

        if (blockSize >=   2)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version is completely unrolled.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads.  This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce5_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T myMax = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        myMax = max(myMax,g_idata[i+blockSize]);

    sdata[tid] = myMax;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + 256]);
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid +  64]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = myMax = max(myMax, smem[tid + 32]);
        }

        if (blockSize >=  32)
        {
            smem[tid] = myMax = max(myMax, smem[tid + 16]);
        }

        if (blockSize >=  16)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  8]);
        }

        if (blockSize >=   8)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  4]);
        }

        if (blockSize >=   4)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  2]);
        }

        if (blockSize >=   2)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_max(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T myMax = (i < n) ? g_idata[i] : 0;
    

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myMax = max(myMax,g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myMax = max(myMax,g_idata[i+blockSize]);

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myMax;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = myMax = max(myMax,sdata[tid + 256]);
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = myMax = max(myMax,sdata[tid + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid +  64]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = myMax = max(myMax, smem[tid + 32]);
        }

        if (blockSize >=  32)
        {
            smem[tid] = myMax = max(myMax, smem[tid + 16]);
        }

        if (blockSize >=  16)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  8]);
        }

        if (blockSize >=   8)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  4]);
        }

        if (blockSize >=   4)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  2]);
        }

        if (blockSize >=   2)
        {
            smem[tid] = myMax = max(myMax, smem[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}




////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce_max(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata, cudaStream_t stream)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
        case 0:
            reduce0_max<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size);
            break;
        case 1:
            reduce1_max<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size);
            break;
        case 2:
            reduce2_max<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size);
            break;
        case 3:
            reduce3_max<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size);
            break;
        case 4:
            switch (threads)
            {
                case 512:
                    reduce4_max<T, 512><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 256:
                    reduce4_max<T, 256><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 128:
                    reduce4_max<T, 128><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 64:
                    reduce4_max<T,  64><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 32:
                    reduce4_max<T,  32><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 16:
                    reduce4_max<T,  16><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  8:
                    reduce4_max<T,   8><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  4:
                    reduce4_max<T,   4><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  2:
                    reduce4_max<T,   2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  1:
                    reduce4_max<T,   1><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
            }
            break;

        case 5:
            switch (threads)
            {
                case 512:
                    reduce5_max<T, 512><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 256:
                    reduce5_max<T, 256><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 128:
                    reduce5_max<T, 128><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 64:
                    reduce5_max<T,  64><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 32:
                    reduce5_max<T,  32><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case 16:
                    reduce5_max<T,  16><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  8:
                    reduce5_max<T,   8><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  4:
                    reduce5_max<T,   4><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  2:
                    reduce5_max<T,   2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                case  1:
                    reduce5_max<T,   1><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
            }
            break;

        case 6:
        default:
            if (isPow2(size))
            {
                switch (threads)
                {
                    case 512:
                        reduce6_max<T, 512, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 256:
                        reduce6_max<T, 256, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 128:
                        reduce6_max<T, 128, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 64:
                        reduce6_max<T,  64, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 32:
                        reduce6_max<T,  32, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 16:
                        reduce6_max<T,  16, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  8:
                        reduce6_max<T,   8, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  4:
                        reduce6_max<T,   4, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  2:
                        reduce6_max<T,   2, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  1:
                        reduce6_max<T,   1, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                }
            }
            else
            {
                switch (threads)
                {
                    case 512:
                        reduce6_max<T, 512, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 256:
                        reduce6_max<T, 256, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 128:
                        reduce6_max<T, 128, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 64:
                        reduce6_max<T,  64, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 32:
                        reduce6_max<T,  32, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case 16:
                        reduce6_max<T,  16, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  8:
                        reduce6_max<T,   8, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  4:
                        reduce6_max<T,   4, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  2:
                        reduce6_max<T,   2, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                    case  1:
                        reduce6_max<T,   1, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); break;
                }
            }
            break;
    }
}

// Instantiate the reduction function for 3 types
template void
reduce_max<int>(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata, cudaStream_t stream);

template void
reduce_max<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata, cudaStream_t stream);

template void
reduce_max<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////



//functions to call this stuff
template<class T>
void maxIntoBlocks(T * devPointer_i,T * devPointer_o, int &nb, const long int M, cudaStream_t stream) {
    int whichKernel = 6;
    int maxBlocks  = 1024;
    int maxThreads = 512;
    int numThreads = 512;

    getNumBlocksAndThreads(whichKernel,M, maxBlocks, maxThreads,nb, numThreads);

    //mexPrintf("nBlocks %d, nT %d\n",numBlocks[ii],numThreads);
    reduce_max<T>((int)M, numThreads, nb, whichKernel, devPointer_i, devPointer_o, stream);
}
// Instantiate the reduction function for 3 types
template void
maxIntoBlocks<int>(int  * devPointer_i,int * devPointer_o, int &nb, const long int M, cudaStream_t stream);

template void
maxIntoBlocks<float>(float  * devPointer_i,float * devPointer_o, int &nb, const long int M, cudaStream_t stream);

template void
maxIntoBlocks<double>(double * devPointer_i,double * devPointer_o, int &nb, const long int M, cudaStream_t stream);


#endif // #ifndef _REDUCE_KERNEL_H_
#endif

