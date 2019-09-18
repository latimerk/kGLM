#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include <curand.h>

#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"


/*
 * 0 = matrix A (m x n)
 * 1 = vector b (m x 1)
 * 2 = is vector in gpu? 0 for false, by default is false
 * 3 = (optional) 1 by n output vector on gpu
 *output, lhs
 * 0 = (optional) b'*A (1 by n);
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    //mexPrintf("init (M)");
    int devNum = kcGetArrayDev(prhs[0]);
    checkCudaErrors(cudaSetDevice(devNum));
    
    long int M = kcGetArraySize(prhs[0], 0);
    long int N = kcGetArraySize(prhs[0], 1);
    //mexPrintf("%d rows, %d columns\n",M,N);
    void * A = kcGetArrayData(prhs[0]);
    
    int arrayType = kcGetArrayType(prhs[0]);
    if(arrayType != KC_DOUBLE_ARRAY && arrayType != KC_FLOAT_ARRAY) {
        mexErrMsgTxt("Invalid matrix input type.\n");
    }
    
    
    cublasOperation_t opt = CUBLAS_OP_N;
    
    void * b_gpu;
    int freeB = 0;
    if(nrhs < 3 || mxGetScalar(prhs[2]) <= 0) {
        //mexPrintf("moving b to device\n");
        long int M2 = mxGetNumberOfElements(prhs[1]);
        if(M2 != N) {
            mexErrMsgTxt("Invalid array input size.\n");
        }
        
        double * b = (double *)mxGetPr(prhs[1]);
        if(arrayType == KC_DOUBLE_ARRAY) {
            checkCudaErrors(cudaMalloc((void**)&b_gpu,sizeof(double)*N));
            checkCudaErrors(cudaMemcpy(b_gpu,b,sizeof(double)*N,cudaMemcpyHostToDevice));
        }
        else {
            float * b2 = (float*)malloc(N*sizeof(float));
            for(int ii = 0; ii < N; ii++) {
                b2[ii] = (float)b[ii];
            }
            checkCudaErrors(cudaMalloc((void**)&b_gpu,sizeof(float)*N));
            checkCudaErrors(cudaMemcpy(b_gpu,b2,sizeof(float)*N,cudaMemcpyHostToDevice));
            free(b2);
        }
        freeB = 1;
    }
    else {
        long int M2 = kcGetArrayNumEl(prhs[1]);

        if(M2 != N) {
            mexErrMsgTxt("Invalid array input size.\n");
        }
        int arrayType2 = kcGetArrayType(prhs[1]);
        if(arrayType != arrayType2) {
            mexErrMsgTxt("Floating point types do not match");
        }
        
        b_gpu = kcGetArrayData(prhs[1]);
    }
    
    void * R;
    int freeR;
    if(nrhs <= 3) {
        freeR = 1;
        if(arrayType == KC_DOUBLE_ARRAY) {
            checkCudaErrors(cudaMalloc((void**)&R,sizeof(double)*1*M));
        }
        else {
            checkCudaErrors(cudaMalloc((void**)&R,sizeof(float)*1*M));
        }
    }
    else {
        if(kcGetArrayNumEl(prhs[3]) < M) {
            mexErrMsgTxt("Output array input size.\n");
        }
        int arrayType3 = kcGetArrayType(prhs[3]);
        if(arrayType != arrayType3) {
            mexErrMsgTxt("Floating point types do not match");
        }
        R = kcGetArrayData(prhs[3]);
        freeR = 0;
    }
    
    //setup cublas
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        if(freeB) {
            checkCudaErrors(cudaFree(b_gpu));
        }
        checkCudaErrors(cudaFree(R));
        mexErrMsgTxt("CUBLAS initialization failed\n");
    }
    
    
    //multiplies A'*b
    if(arrayType == KC_DOUBLE_ARRAY) {
        double alpha = (double)1.0;
        double beta = (double)0;
        stat =  cublasGEMV(handle,  opt,
                               M, N,
                               &alpha,
                               (double*)A, M,
                               (double*)b_gpu, 1,
                               &beta,
                               (double*)R, 1);
    }
    else {
        float alpha = (float)1.0;
        float beta = (float)0;
        stat =  cublasGEMV(handle,  opt,
                               M, N,
                               &alpha,
                               (float*)A, M,
                               (float*)b_gpu, 1,
                               &beta,
                               (float*)R, 1);
    }
    if(stat != CUBLAS_STATUS_SUCCESS) {
        if(freeB) {
            checkCudaErrors(cudaFree(b_gpu));
        }
        checkCudaErrors(cudaFree(R));
        cublasDestroy(handle);
        mexErrMsgTxt("matrix multiplication failed");
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    //gets result into host memory
    if(nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(M,1,mxDOUBLE_CLASS,mxREAL);

        if(arrayType == KC_DOUBLE_ARRAY) {
            stat = cublasGetMatrix (M, 1, sizeof (double), R, M, (double*)mxGetPr(plhs[0]), M);
        }
        else {
            float * ans_host = (float*)malloc(N*sizeof(float));
            stat = cublasGetMatrix (M, 1, sizeof (float), R, M, ans_host, M);
            double * ans_matlab = (double*)mxGetPr(plhs[0]);
            for(int ii = 0; ii < M; ii++) {
                ans_matlab[ii] = ans_host[ii];
            }
            free(ans_host);
        }
        if(stat != CUBLAS_STATUS_SUCCESS) {
            if(freeB) {
                checkCudaErrors(cudaFree(b_gpu));
            }
            checkCudaErrors(cudaFree(R));
            cublasDestroy(handle);
            mexErrMsgTxt("result upload failed");
        }
    }

    
    //clear memory
    cublasDestroy(handle);
    if(freeB) {
        checkCudaErrors(cudaFree(b_gpu));
    }
    if(freeR) {
        checkCudaErrors(cudaFree(R));
    }
    checkCudaErrors(cudaDeviceSynchronize());
    //mexPrintf("end (M)\n");
}
