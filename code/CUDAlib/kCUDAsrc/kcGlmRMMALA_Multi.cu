#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <curand.h>

#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"
#include "reduction.h"
#include "reduction_max.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"
#include "kcMatrixMultHelper.h"


void checkCUsolverStatus(cusolverStatus_t status, const char * str) {
    switch(status) {
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            mexPrintf("CUSOLVER_STATUS_NOT_INITIALIZED error: %s\n",str);
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            mexPrintf("CUSOLVER_STATUS_ALLOC_FAILED error: %s\n",str);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            mexPrintf("CUSOLVER_STATUS_INVALID_VALUE error: %s\n",str);
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            mexPrintf("CUSOLVER_STATUS_ARCH_MISMATCH error: %s\n",str);
            break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            mexPrintf("CUSOLVER_STATUS_EXECUTION_FAILED error: %s\n",str);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            mexPrintf("CUSOLVER_STATUS_INTERNAL_ERROR error: %s\n",str);
            break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            mexPrintf("CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED error: %s\n",str);
            break;
            
    }
}

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Structs                       ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T> 
struct GPUGLMplan {
    //Host-side input data
    long int M; //rows of X
    long int N; //columns of X
    
    // GLM inputs (on GPU)
    T * X;
    T * y;
    T * b;
    
    // compute space (on GPU)
    T * C;
    T * C2;
    T * F;  //column of C
    T * Fs; //column of C
    T * G;  //column of C
    T * Xb; //column of C
    
    //answer space
    T * H_ans; //N*(N+2)
    T * G_ans; //N*1
    T * M_ans; //N*N
    T * M2_ans; //N*1

    //answer space
    T * M2_ans_host; //N*1
    T * H_ans_host; //N*(N)
    T * G_ans_host; //N*1
    T * F_ans_host; //M
    
    
    T * Workspace;
    //int * devIpiv;
    int Lwork;
    
    unsigned int devNum;
    
    int * devInfo;
    int devInfo_host;
    
    int useXb;

    
    //Stream for asynchronous command execution
    cudaStream_t stream;
    
    cublasHandle_t handle;
    cusolverDnHandle_t solverHandle;

};


//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Setup Plan                    ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T>
void setupPlan(GPUGLMplan<T> &plan, const mxArray * X, const mxArray * y, const mxArray * C, const mxArray * b, const mxArray * Xb, const int useXb) {

    plan.devNum = kcGetArrayDev(X);
    
    if(kcGetArrayDev(y) != plan.devNum || kcGetArrayDev(C) != plan.devNum) {
        mexErrMsgTxt("Arrays not all placed on proper devices!\n");
    }
    
    checkCudaErrors(cudaSetDevice(plan.devNum));
    checkCudaErrors(cudaStreamCreate(&(plan.stream)));
    
    
    plan.M = kcGetArraySize(X, 0);
    plan.N = kcGetArraySize(X, 1);
    
    plan.useXb = useXb;
    
    //checks input sizes
    long int N_C = kcGetArraySize(C, 1);
    long int M_C = kcGetArraySize(C, 0);
    if(plan.M * (plan.N + 4) > N_C*M_C) {
        mexPrintf("Insufficient computation space for block on device %d! (sizes X=(%d,%d) C=(%d,%d))\n",plan.devNum,plan.M,plan.N,M_C,N_C);
        mexErrMsgTxt("Invalid input!");
    }
    long int N_Y = kcGetArraySize(y, 1);
    long int M_Y = kcGetArraySize(y, 0);
    if(plan.M != N_Y*M_Y) {
        mexPrintf("Number of observations does not match rows in design matrix on block on device %d! (sizes X=(%d,%d) Y=(%d,%d))\n",plan.devNum,plan.M,plan.N,M_Y,N_Y);
        mexErrMsgTxt("Invalid input!");
    }
    long int N_B = mxGetNumberOfElements(b);
    if(plan.N != N_B) {
        mexPrintf("Number of columns in design matrix does not match weight vector length on block on device %d! (sizes X=(%d,%d) B=(%d))\n",plan.devNum,plan.M,plan.N,N_B);
        mexErrMsgTxt("Invalid input!");
    }
    if(kcGetArrayType(X) != getKCtype<T>()) {
        mexPrintf("X is not the correct floating point type.");
        mexErrMsgTxt("Invalid input!");
    }
    if(kcGetArrayType(y) != getKCtype<T>()) {
        mexPrintf("y is not the correct floating point type.");
        mexErrMsgTxt("Invalid input!");
    }
    if(kcGetArrayType(C) != getKCtype<T>()) {
        mexPrintf("C is not the correct floating point type.");
        mexErrMsgTxt("Invalid input!");
    }
    
    
    
    //mexPrintf("Dev %d, M = %d, N = %d\n", plan.devNum, plan.M, plan.N);
    
    plan.X = (T*)kcGetArrayData(X);
    plan.y = (T*)kcGetArrayData(y);
    plan.C = (T*)kcGetArrayData(C);
    
    
    //plan.C2 = plan.C+plan.M*(plan.N+4);
    plan.G  = plan.C+plan.M*(plan.N+0);
    plan.F  = plan.C+plan.M*(plan.N+1);
    plan.Fs = plan.C+plan.M*(plan.N+2);
    
    plan.Xb = plan.C+plan.M*(plan.N+3);
    if(plan.useXb) {
        if(kcGetArrayDev(Xb) != plan.devNum ) {
            mexErrMsgTxt("Arrays not all placed on proper devices!\n");
        }
        if(kcGetArrayType(Xb) != getKCtype<T>()) {
            mexPrintf("C is not the correct floating point type.");
            mexErrMsgTxt("Invalid input!");
        }
        
        long int N_Xb = kcGetArraySize(Xb, 1);
        long int M_Xb = kcGetArraySize(Xb, 0);
        if(plan.M != N_Xb * M_Xb) {
            mexPrintf("Number of constant rate mods does not match rows in design matrix on block on device %d! (sizes X=(%d,%d) Y=(%d,%d))\n",plan.devNum,plan.M,plan.N,M_Xb,N_Xb);
            mexErrMsgTxt("Invalid input!");
        }
        checkCudaErrors(cudaMemcpyAsync(plan.Xb,(T *) kcGetArrayData(Xb), sizeof(T)*plan.M,cudaMemcpyDeviceToDevice,plan.stream));
    }
    
    plan.M2_ans_host = (T*)malloc(plan.N*sizeof(T));
    plan.H_ans_host = (T*)malloc(plan.N*plan.N*sizeof(T));
    plan.G_ans_host = (T*)malloc(plan.N*sizeof(T));
    plan.F_ans_host = (T*)malloc(plan.M*sizeof(T));
    
    //checkCudaErrors(cudaMalloc((void**)&(plan.devInfo),sizeof(int)));
    checkCudaErrors(cudaHostRegister(&(plan.devInfo_host),sizeof(int),cudaHostRegisterMapped));
    //cudaHostRegister(&(plan.devInfo_host),sizeof(int),cudaHostRegisterMapped); cudaGetLastError();
    checkCudaErrors(cudaHostGetDevicePointer((void**)&(plan.devInfo),&(plan.devInfo_host),0));
    
    //checkCudaErrors(cudaMalloc((void**)&(plan.devIpiv),sizeof(int)*plan.N));
    
    checkCudaErrors(cudaMalloc((void**)&(plan.b),sizeof(T)*plan.N));
    T * b_f = (T*)malloc(plan.N*sizeof(T));
    double *   b_p = (double *)mxGetPr(b);
    for(int ii = 0; ii < plan.N; ii++) {
        b_f[ii] = (T)(b_p[ii]);
    }
    checkCudaErrors(cudaMemcpyAsync((plan.b),b_f,  sizeof(T)*plan.N,cudaMemcpyHostToDevice,plan.stream));
    
    checkCudaErrors(cudaMalloc((void**)&(plan.H_ans),sizeof(T)*(plan.N*2 + 2)*plan.N));
    plan.G_ans = plan.H_ans+plan.N*(plan.N+0);
    plan.M_ans = plan.H_ans+plan.N*(plan.N+1);
    plan.M2_ans = plan.H_ans+plan.N*(plan.N*2+1);
    
    cublasStatus_t stat;
    stat = cublasCreate(&(plan.handle));
    if(stat != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS initialization failed\n");
    }
    
    checkCUsolverStatus(cusolverDnCreate(&plan.solverHandle),"creating handle");
    cudaStreamSynchronize(plan.stream);
    free(b_f);
    //cudaStreamSynchronize(plan.stream);
}
template void setupPlan<float>(GPUGLMplan<float> &plan, const mxArray * X, const mxArray * y, const mxArray * C, const mxArray * b, const mxArray * Xb, const int useXb);
template void setupPlan<double>(GPUGLMplan<double> &plan, const mxArray * X, const mxArray * y, const mxArray * C, const mxArray * b, const mxArray * Xb, const int useXb);



//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Destroy Plan                   ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T>    
void destroyPlan(GPUGLMplan<T> &plan) {
    checkCudaErrors(cudaSetDevice(plan.devNum));
    
    
    //mexPrintf("Dev %d, M = %d, N = %d\n", plan.devNum, plan.M, plan.N);
    cublasDestroy(plan.handle);
    checkCUsolverStatus(cusolverDnDestroy(plan.solverHandle),"destroying handle");
    checkCudaErrors(cudaFree(plan.b));
    checkCudaErrors(cudaFree(plan.H_ans));
    //checkCudaErrors(cudaFree(plan.devInfo));
    checkCudaErrors(cudaHostUnregister(&(plan.devInfo_host)));
    
    //checkCudaErrors(cudaFree(plan.devIpiv));
    
    free(plan.H_ans_host);
    free(plan.M2_ans_host);
    free(plan.G_ans_host);
    free(plan.F_ans_host);
    checkCudaErrors(cudaStreamDestroy(plan.stream));
}
template void  destroyPlan<float>(GPUGLMplan<float> &plan);
template void  destroyPlan<double>(GPUGLMplan<double> &plan);

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Poisson                  ==============================================
//==========================================                    Link  : Exp                      ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
__global__ void d2llKernel_ExpPoisson(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T l, const long int M, const long int N) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T cXb  = Xb[row];
            T ceXb = exp(cXb-l);
            T ceXb_12 = exp((cXb-l)/T(2)); // sqrt of the Hessian term, used to compute Hessian as C'*C so that it's Hermetian. C = bsxfun(@times,X,ceXb_12)
                                                                // l is some term I added in an attempt to increase numerical stability, could be probably be set to 0 
            
            C[row  + M*col] = X[row + M*col] *ceXb_12;
            //C2[col + N*row] = X[row + M*col] *ceXb;//C[row + M*col];
            
            if(col == 0) {
                T ceXb_0 = exp(cXb);
                
                T cy  = y[row];
                f[row]   = cy*cXb         - ceXb_0;
                g[row]   = cy*exp(-l)     - ceXb;
            }
        }
    }
}
template __global__ void d2llKernel_ExpPoisson<float>(float * C, float * f, float * g,const float * X, const float * y, const float * Xb, const float l, const long int M, const long int N);
template __global__ void d2llKernel_ExpPoisson<double>(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double l, const long int M, const long int N);

template <class T>
__global__ void d2llKernel_ExpPoisson2(T * C, const T * X, const T * Xb, const T l, const long int M, const long int N) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T cXb  = Xb[row];
            T ceXb = exp(cXb-l); // Hessian term
            C[col + N*row] = X[row + M*col] *ceXb;//C[row + M*col];
        }
    }
}
template __global__ void d2llKernel_ExpPoisson2<float>(float * C, const float * X, const float * Xb, const float l, const long int M, const long int N) ;
template __global__ void d2llKernel_ExpPoisson2<double>(double * C, const double * X, const double * Xb, const double l, const long int M, const long int N) ;

template <class T>
__global__ void hadamardProdTransposeC2(T * C, const T * X, const T * Xb, const T l, const long int M, const long int N) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            //double cXb  = Xb[row];
            T ceXb = 1.0;//exp(min(KC_EXP_MAX,(cXb-l)));
            C[col  + N*row]  *= X[row + M*col] *ceXb;
            //C[row  + M*col]
            
        }
    }
}
template __global__ void hadamardProdTransposeC2<float>(float * C, const float * X, const float * Xb, const float l, const long int M, const long int N);
template __global__ void hadamardProdTransposeC2<double>(double * C, const double * X, const double * Xb, const double l, const long int M, const long int N);

template <class T>
__global__ void hadamardProdT(T * C, const T * X,  const long int M, const long int N) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            C[col  + N*row]  *= X[row + M*col] ;
        }
    }
}
template __global__ void hadamardProdT<float>(float * C, const float * X,  const long int M, const long int N);
template __global__ void hadamardProdT<double>(double * C, const double * X,  const long int M, const long int N);

template <class T>
__global__ void setOnes(T * X, const long int N) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    if(row < N) {
        X[row]  = 1;
    }
}
template __global__ void setOnes<float>(float * X, const long int N);
template __global__ void setOnes<double>(double * X, const long int N);

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       solvers                       ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
cusolverStatus_t getRFbufferSize(cusolverDnHandle_t handle, int a, int b, double * A, int lda, int * bufferSize) {
    return cusolverDnDgetrf_bufferSize(handle,a,b,A,lda,bufferSize);
}
cusolverStatus_t getRFbufferSize(cusolverDnHandle_t handle, int a, int b, float * A, int lda, int * bufferSize) {
    return cusolverDnSgetrf_bufferSize(handle,a,b,A,lda,bufferSize);
}

cusolverStatus_t getRFsolver(cusolverDnHandle_t handle, int a, int b, double * A, int lda, double * buffer, int * ipiv, int * info) {
    return cusolverDnDgetrf(handle, a, b, A, lda, buffer, ipiv, info);
}
cusolverStatus_t getRFsolver(cusolverDnHandle_t handle, int a, int b, float * A, int lda, float * buffer, int * ipiv, int * info) {
    return cusolverDnSgetrf(handle, a, b, A, lda, buffer, ipiv, info);
}

cusolverStatus_t getRSsolver(cusolverDnHandle_t handle, cublasOperation_t op, int a, int b, double * A, int lda, int * ipiv, double * B, int c, int * info) {
    return cusolverDnDgetrs(handle, op, a, b, A, lda, ipiv, B, c, info);
}
cusolverStatus_t getRSsolver(cusolverDnHandle_t handle, cublasOperation_t op, int a, int b, float * A, int lda, int * ipiv, float * B, int c, int * info) {
    return cusolverDnSgetrs(handle, op, a, b, A, lda, ipiv, B, c, info);
}


template <class T>
int linearSolverLU(cusolverDnHandle_t handle, int n, int m, T *A, int lda, T *b, int * info, T * buffer, int * ipiv) {
    
    cusolverStatus_t solverStatus;
    int bufferSize = 0;

    checkCudaErrors(getRFbufferSize(handle, n, n, A, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&buffer, sizeof(T)*bufferSize));


    // prepare a copy of A because getrf will overwrite A with L


    checkCudaErrors(getRFsolver(handle, n, n, A, lda, buffer, ipiv, info));

    solverStatus = getRSsolver(handle, CUBLAS_OP_N, n, m, A, lda, ipiv, b, n, info);
    checkCUsolverStatus(solverStatus, "at cusolverDnDgetrs");
    
    return 0;
}
template int linearSolverLU<float>(cusolverDnHandle_t handle, int n, int m, float *A, int lda, float *b, int * info, float * buffer, int * ipiv);
template int linearSolverLU<double>(cusolverDnHandle_t handle, int n, int m, double *A, int lda, double *b, int * info, double * buffer, int * ipiv);


cusolverStatus_t getPOTRFbufferSize(cusolverDnHandle_t handle,cublasFillMode_t uplo,int a, double * A, int lda, int * bufferSize) {
    return cusolverDnDpotrf_bufferSize(handle, uplo, a, A, lda, bufferSize);
}
cusolverStatus_t getPOTRFbufferSize(cusolverDnHandle_t handle,cublasFillMode_t uplo,int a, float * A, int lda, int * bufferSize) {
    return cusolverDnSpotrf_bufferSize(handle, uplo, a, A, lda, bufferSize);
}

cusolverStatus_t getPOTRFsolver(cusolverDnHandle_t handle,cublasFillMode_t uplo,int a, double * A, int lda, double * buffer, int bufferSize, int * info) {
    return cusolverDnDpotrf(handle, uplo, a, A, lda, buffer, bufferSize, info);
}
cusolverStatus_t getPOTRFsolver(cusolverDnHandle_t handle,cublasFillMode_t uplo,int a, float * A, int lda, float * buffer, int bufferSize, int * info) {
    return cusolverDnSpotrf(handle, uplo, a, A, lda, buffer, bufferSize, info);
}

cusolverStatus_t getPOTRSsolver(cusolverDnHandle_t handle,cublasFillMode_t uplo,int a, int b, double * A, int lda, double * B, int c, int * info) {
    return cusolverDnDpotrs(handle, uplo, a, b, A, lda, B, c, info);
}
cusolverStatus_t getPOTRSsolver(cusolverDnHandle_t handle,cublasFillMode_t uplo,int a, int b, float * A, int lda, float * B, int c, int * info) {
    return cusolverDnSpotrs(handle, uplo, a, b, A, lda, B, c, info);
}

template <class T>
int linearSolverCHOL(cusolverDnHandle_t handle, int n, int m, T *A, int lda, T *b, int * info, T * buffer) {
    int bufferSize = 0;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(getPOTRFbufferSize(handle, uplo, n, A, lda, &bufferSize));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(T)*bufferSize));


    // prepare a copy of A because potrf will overwrite A with L


    checkCudaErrors(getPOTRFsolver(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(getPOTRSsolver(handle, uplo, n, m, A, lda, b, n, info));

    if (buffer) { checkCudaErrors(cudaFree(buffer)); }

    return 0;
}
template int linearSolverCHOL<float>(cusolverDnHandle_t handle, int n, int m, float *A, int lda, float *b, int * info,float * buffer);
template int linearSolverCHOL<double>(cusolverDnHandle_t handle, int n, int m, double *A, int lda, double *b, int * info, double * buffer);




//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       computeFGH                    ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================


template <class T> 
void computeFGH(GPUGLMplan<T> * plan, double * f, double * g, double * h, double * M, double * M2, double * H_prior, const int useXb, const long int N, const int nGPUs) {
    cublasStatus_t stat;
    
    dim3 block_size;
    dim3 block_size2;
    block_size.x = 256;
    block_size.y = 1024/block_size.x;
    
    T alpha = (T)1.0;
    T * l_s = (T*)malloc(nGPUs*sizeof(T));
    T * h_T = (T*)malloc(N*N*sizeof(T));
    T beta  = (T)0.0;
    T beta_Xb  = (T)0.0;
    if(useXb) {
        //mexPrintf("useXb\n");
        beta_Xb  = (T)1.0;
    }
    long int ii,jj;

    //clock_t begin, end;
    //double time_spent;
    
    int * numBlocks = (int *)malloc(nGPUs*sizeof(int));
    
    for(ii = 0; ii < nGPUs; ii++) {
        
        //mexPrintf("running comp loop plan %d\n",ii);
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cublasSetStream(plan[ii].handle, plan[ii].stream);

        //mexPrintf("  X*b \n");
        //begin = clock();
        stat =  cublasGEMV(plan[ii].handle,  CUBLAS_OP_N,
                           plan[ii].M, plan[ii].N,
                           &alpha,
                           plan[ii].X, plan[ii].M,
                           plan[ii].b, 1,
                           &beta_Xb,
                           plan[ii].Xb, 1);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("initial matrix times filter failed");
        }
        
        int nb = 0;
        maxIntoBlocks<T>(plan[ii].Xb,plan[ii].F, nb, plan[ii].M,  plan[ii].stream); //in reduction_max.h 
        numBlocks[ii] = nb;     
        
        checkCudaErrors(cudaMemcpyAsync(plan[ii].F_ans_host, plan[ii].F,     numBlocks[ii] *sizeof(T), cudaMemcpyDeviceToHost, plan[ii].stream));
    }
    for(ii = 0; ii < nGPUs; ii++) {
      
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cudaStreamSynchronize(plan[ii].stream);
        
        //mexPrintf("  f \n");
        l_s[ii] = plan[ii].F_ans_host[0];
        for(jj = 1; jj < numBlocks[ii]; jj++) {
            l_s[ii] = max(plan[ii].F_ans_host[jj],l_s[ii]);
        }
        //mexPrintf("  l_s %.2f \n",l_s[ii]);
        T cc = 10;
        if(l_s[ii] > cc) {
            l_s[ii] = l_s[ii]+cc;
        }
        else {
            l_s[ii] = 0;
        }
    }
    
    for(ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("running comp loop plan %d\n",ii);
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cublasSetStream(plan[ii].handle, plan[ii].stream);
        
        // configure a two dimensional grid
        dim3 grid_size;
        grid_size.x = plan[ii].M/block_size.x + ((plan[ii].M%block_size.x == 0)? 0:1);
        grid_size.y = plan[ii].N/block_size.y + ((plan[ii].N%block_size.y == 0)? 0:1);

        T l_s_c  = l_s[ii]; //l_s[ii] could (should?) probably be set to 0. I added it in an attempt to increase numerical stability by not exponentiating huge numbers
        T el_s_c = exp(l_s_c);
        
        //mexPrintf("  multiply rows\n");                     //(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N) {
        d2llKernel_ExpPoisson<T><<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].C,plan[ii].F,plan[ii].G,plan[ii].X, plan[ii].y, plan[ii].Xb, l_s_c, plan[ii].M, plan[ii].N);
        getLastCudaError("d2llKernel() execution failed.\n");

        

        //multiplies C'*C, result in H
        //mexPrintf("  C*C \n");
        //cublasSetStream(plan[ii].handle, plan[ii].stream);
        
        
        matrixMultiplyHelper(plan[ii].C,plan[ii].C,plan[ii].H_ans,N,N,plan[ii].M,&el_s_c,&beta,plan[ii].stream,plan[ii].handle);
        stat =  cublasGEMM(plan[ii].handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           N, 1, plan[ii].M,
                           &el_s_c,
                           plan[ii].X, plan[ii].M,
                           plan[ii].G, plan[ii].M,
                           &beta,
                           plan[ii].G_ans, N);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("gradient matrix times failed");
        }
        
        
        int nb = 0;
        sumIntoBlocks<T>(plan[ii].F,plan[ii].Fs, nb, plan[ii].M,  plan[ii].stream); //in reduction.h
        numBlocks[ii] = nb;
        
        
        d2llKernel_ExpPoisson2<T><<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].C,plan[ii].X, plan[ii].Xb, l_s_c, plan[ii].M, plan[ii].N);
        getLastCudaError("d2llKernel2() execution failed.\n");
        
        //cudaStreamSynchronize(plan[ii].stream); end = clock(); time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC); mexPrintf("B1 %d = %2.4f\n",ii,time_spent);
    }        
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        //cublasSetStream(plan[ii].handle, plan[ii].stream);
	    
        checkCudaErrors(cudaMemcpyAsync(plan[ii].H_ans_host, plan[ii].H_ans, N*N *sizeof(T),          cudaMemcpyDeviceToHost, plan[ii].stream));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].G_ans_host, plan[ii].G_ans, N*1 *sizeof(T),          cudaMemcpyDeviceToHost, plan[ii].stream));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].F_ans_host, plan[ii].Fs,     numBlocks[ii] *sizeof(T), cudaMemcpyDeviceToHost, plan[ii].stream));
	
        
    }
                
    for(ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("running sum loop plan %d, f = %2.2f\n",ii,f[0]);
        //mexPrintf("  M = %d, N = %d, dt=%2.2f\n", plan[ii].M, plan[ii].N,plan[ii].dt);
        //begin = clock();
        
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cudaStreamSynchronize(plan[ii].stream);
        
        //mexPrintf("  f \n");
        double f_p = 0;
        for(jj = 0; jj < numBlocks[ii]; jj++) {
            f_p += plan[ii].F_ans_host[jj];
        }
        if(ii == 0) {
            f[0]  = 0;
        }
        f[0] += f_p;
        //mexPrintf(" f_p = %3.2f\n",f_p);
        
        //mexPrintf("  g \n");
        for(jj = 0; jj < N; jj++) {
            if(ii == 0) {
                g[jj]  = plan[ii].G_ans_host[jj];
            }
            else {
                g[jj] += plan[ii].G_ans_host[jj];
            }
        }
        
        //mexPrintf("  h \n");
        for(jj = 0; jj < N*N; jj++) {
            if(ii == 0) {
                h[jj]  = plan[ii].H_ans_host[jj] + H_prior[jj];
            }
            else {
                h[jj] += plan[ii].H_ans_host[jj];
            }
        }
        
        //end = clock();
        //time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        //mexPrintf("FB %d = %2.4f\n",ii,time_spent);
    }
    for(jj = 0; jj < N*N; jj++) {
        h_T[jj] = (T)h[jj];
    }
    free(numBlocks);
    
   
    
    
    //compute the RMMALA mean
    //G = local variable h
    cusolverStatus_t solverStatus;
    block_size.x = 128;
    block_size.y = 1024/block_size.x;
    block_size.x = 64;
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cublasSetStream(plan[ii].handle, plan[ii].stream);
        checkCUsolverStatus(cusolverDnSetStream(plan[ii].solverHandle, plan[ii].stream),"at cusolverDnSetStream");
        
        //copy G to GPU
        checkCudaErrors(cudaMemcpyAsync(plan[ii].M_ans, h_T, N*N *sizeof(T),          cudaMemcpyHostToDevice, plan[ii].stream));
        
        //solve G\C2 -> C2 (which is size of transpose(C))
        int USE_SOLVER_1 = 1;
        if(USE_SOLVER_1) {
            
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER; //CUBLAS_FILL_MODE_UPPER  CUBLAS_FILL_MODE_LOWER
            solverStatus = getPOTRFbufferSize(plan[ii].solverHandle, //cusolverDnDpotrf_bufferSize
                     uplo,
                     plan[ii].N,
                     plan[ii].M_ans,
                     plan[ii].N,
                     &(plan[ii].Lwork) );
            checkCUsolverStatus(solverStatus, "at cusolverDnDpotrf_bufferSize");
            
            //mexPrintf("allocating workspace of %d\n",plan[ii].Lwork);
            checkCudaErrors(cudaMalloc((void**)&(plan[ii].Workspace),sizeof(T)*plan[ii].Lwork));
            //int info_gpu;
            solverStatus = getPOTRFsolver(plan[ii].solverHandle, //cusolverDnDpotrf
                     uplo,
                     plan[ii].N,
                     plan[ii].M_ans,
                     plan[ii].N,
                     plan[ii].Workspace,
                     plan[ii].Lwork,
                     plan[ii].devInfo );
            /*checkCudaErrors(cudaMemcpyAsync(&info_gpu, plan[ii].devInfo, sizeof(int), cudaMemcpyDeviceToHost, plan[ii].stream));
            if(plan[ii].devInfo_host != 0) {
                mexPrintf("cusolverDnDpotrf warning: %d on thread %d\n",plan[ii].devInfo_host,ii);
            }*/
            checkCUsolverStatus(solverStatus, "at cusolverDnDpotrf");
            
            //begin = clock();
            solverStatus = getPOTRSsolver(plan[ii].solverHandle,//cusolverDnDpotrs
                     uplo,
                     plan[ii].N,
                     plan[ii].M,
                     plan[ii].M_ans,
                     plan[ii].N,
                     plan[ii].C,
                     plan[ii].N,
                     plan[ii].devInfo);
            /*checkCudaErrors(cudaMemcpyAsync(&info_gpu, plan[ii].devInfo, sizeof(int), cudaMemcpyDeviceToHost, plan[ii].stream));
            if(plan[ii].devInfo_host != 0) {
                mexPrintf("cusolverDnDpotrs warning: %d on thread %d\n",plan[ii].devInfo_host,ii);
            }*/
            checkCUsolverStatus(solverStatus, "at cusolverDnDpotrs");
            //cudaStreamSynchronize(plan[ii].stream); end = clock(); time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC); mexPrintf("SOLVER3 %d = %2.4f\n",ii,time_spent);
        }
        else {
            linearSolverCHOL<T>(plan[ii].solverHandle, plan[ii].N, plan[ii].M, plan[ii].M_ans, plan[ii].N, plan[ii].C,plan[ii].devInfo,plan[ii].Workspace);
            //linearSolverLU(plan[ii].solverHandle, plan[ii].N, plan[ii].M, plan[ii].M_ans, plan[ii].N, plan[ii].C2,plan[ii].devInfo,plan[ii].Workspace,plan[ii].devIpiv);
            
            if(plan[ii].devInfo_host != 0) {
                mexPrintf("cusolverDnDpotrs warning: %d on thread %d\n",plan[ii].devInfo_host,ii);
            }
            mexPrintf(cudaGetErrorString(cudaGetLastError())); mexPrintf("\n");
            if(ii == 1) {
                mexErrMsgTxt("terminating");
            }
        }
        
        //elementwise multiply C and X -> C
        // configure a two dimensional grid
        dim3 grid_size;
        grid_size.x = plan[ii].M/block_size.x + ((plan[ii].M%block_size.x == 0)? 0:1);
        grid_size.y = plan[ii].N/block_size.y + ((plan[ii].N%block_size.y == 0)? 0:1);
        
        
        double l_s_c  = l_s[ii];
        double el_s_c = exp(l_s_c);
        hadamardProdTransposeC2<T><<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].C, plan[ii].X, plan[ii].Xb, el_s_c, plan[ii].M, plan[ii].N);
        
        //matrix mult C' * X -> plan.M_ans
        //mexPrintf("1) plan[%d].M_ans      = %d\n",ii,(long int)(plan[ii].M_ans));
        //mexPrintf("1) plan[%d].H_ans_host = %d\n",ii,(long int)(plan[ii].H_ans_host));
        matrixMultiplyHelper(plan[ii].C,plan[ii].X,plan[ii].M_ans,N,N,plan[ii].M,&alpha,&beta,plan[ii].stream,plan[ii].handle,CUBLAS_OP_N,CUBLAS_OP_N);
        
        
        hadamardProdT<T><<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].C, plan[ii].X, plan[ii].M, plan[ii].N);
        
        
        dim3 grid_size2;
        grid_size2.x = plan[ii].N/block_size2.x + ((plan[ii].N%block_size2.x == 0)? 0:1);
        setOnes<T><<<grid_size2,block_size2,0,plan[ii].stream>>>(plan[ii].b,plan[ii].N);
                
                
        stat =  cublasGEMV(plan[ii].handle,  CUBLAS_OP_T,
                           plan[ii].N, plan[ii].M,
                           &alpha,
                           plan[ii].C, plan[ii].N,
                           plan[ii].b, 1,
                           &beta,
                           plan[ii].C, 1);
        
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("matrix times vector 1 for tr(GdG) failed");
        }
        
        stat =  cublasGEMV(plan[ii].handle,  CUBLAS_OP_T,
                           plan[ii].M, plan[ii].N,
                           &alpha,
                           plan[ii].X, plan[ii].M,
                           plan[ii].C, 1,
                           &beta,
                           plan[ii].M2_ans, 1);
        
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("matrix times vector 2 for tr(GdG) failed");
        }
    }
    
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        
        //mexPrintf("2) plan[%d].M_ans      = %d\n",ii,(long int)(plan[ii].M_ans));
        //mexPrintf("2) plan[%d].H_ans_host = %d\n",ii,(long int)(plan[ii].H_ans_host));
	    
        //copy M back to host
        checkCudaErrors(cudaMemcpyAsync(plan[ii].H_ans_host , plan[ii].M_ans , plan[ii].N*plan[ii].N *sizeof(T),  cudaMemcpyDeviceToHost, plan[ii].stream));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].M2_ans_host, plan[ii].M2_ans, plan[ii].N*sizeof(T),  cudaMemcpyDeviceToHost, plan[ii].stream));
        
    }
    
    
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cudaStreamSynchronize(plan[ii].stream);
        //free Workspace
        cudaFree(plan[ii].Workspace);
        
        
        double l_s_c  = l_s[ii];
        double el_s_c = exp(l_s_c);
        
        //sum up plan.M_ans_host and M_prior into M
        //mexPrintf("  h \n");
        for(jj = 0; jj < N*N; jj++) {
            if(ii == 0) {
                M[jj]  = el_s_c*plan[ii].H_ans_host[jj];
            }
            else {
                M[jj] += el_s_c*plan[ii].H_ans_host[jj];
            }
        }
        
        
        for(jj = 0; jj < N; jj++) {
            if(ii == 0) {
                M2[jj]  = el_s_c*plan[ii].M2_ans_host[jj];
            }
            else {
                M2[jj] += el_s_c*plan[ii].M2_ans_host[jj];
            }
        }
    }
    free(l_s);
    free(h_T);
}
template void computeFGH<float>(GPUGLMplan<float> * plan, double * f, double * g, double * h, double * M, double * M2, double * H_prior, const int useXb, const long int N, const int nGPUs);
template void computeFGH<double>(GPUGLMplan<double> * plan, double * f, double * g, double * h, double * M, double * M2, double * H_prior, const int useXb, const long int N, const int nGPUs);








//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Main                          ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

/* GPU accellerated helper function for RMMALA for Poisson GLM with exp inverse-link function and a Gaussian prior
 * 0  = filter (w) (m X 1)
 * 1  = inverse of prior covariance matrix(n X n matrix, on cpu)
 * cell array arguments for multi-GPU: (if only using one GPU/block of data, these can be kcArray structs)
 * 2  = matrix X (m x n)
 * 3  = y (m x 1, vector on gpu)
 * 4  = extra space for computations (m x (2n + 3))
 * 5  = (OPTIONAL) (m x 1, vector on GPU) constant rate addition for each trial
 *output, lhs
 * 0 = f; log likelihood
 * 1 = g; dll_db
 * 2 = h; -d2ll_db2 + inverse prior covariance (for Fisher info)
 * 3 = M;  matrix for computing mu - remaing calculations need to be performed in MATLAB to get proposal mean:
 * 4 = M2; vector for computing mu - remaing calculations need to be performed in MATLAB to get proposal mean:
 *
 *mu = w  + e^2/2*(G\g);
 *mu = mu - e^2*sum(G\M,2);
 *mu = mu + e^2/2*sum(inv(G),2).*M2;
 */
template <class T> 
void runRMMALA(double * f,double * g,double * h,double * m,double * m2,double * h_prior,const mxArray * XX,const mxArray * YY,const mxArray * CC,const mxArray * BB,const mxArray * Xb,const int useXb, const int nGPUs) {
    
    GPUGLMplan<T> plans[nGPUs];
    if( mxIsStruct(XX) && nGPUs == 1) {
        setupPlan<T>( plans[0], XX, YY, CC, BB, Xb, useXb);
    }
    else if(mxIsStruct(XX) && nGPUs == 1) {
        mexErrMsgTxt("Multiple GPUs requires cell array inputs.");
    }
    else {
        for(int ii = 0; ii < nGPUs; ii++) {
            mxArray * Xb_c;
            if(useXb) {
                Xb_c = mxGetCell(Xb,ii);
            }
            setupPlan<T>( plans[ii], mxGetCell(XX,ii), mxGetCell(YY,ii), mxGetCell(CC,ii), BB, Xb_c, useXb);
        }
    }
    
    //mexPrintf("computing N=%d\n",N);
    long int N = mxGetNumberOfElements(BB);

    computeFGH<T>(plans, f, g, h, m, m2, h_prior, useXb, N,  nGPUs);
              
    for(int ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("destroying GPU plan %d\n",ii);
        destroyPlan<T>( plans[ii]);
    }
}
template void runRMMALA<float>(double * f,double * g,double * h,double * m,double * m2,double * h_prior,const mxArray * XX,const mxArray * YY,const mxArray * CC,const mxArray * BB,const mxArray * Xb,const int useXb, const int nGPUs);
template void runRMMALA<double>(double * f,double * g,double * h,double * m,double * m2,double * h_prior,const mxArray * XX,const mxArray * YY,const mxArray * CC,const mxArray * BB,const mxArray * Xb,const int useXb, const int nGPUs);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    if((nrhs != 5 && nrhs != 6) || nlhs != 5) {
        mexPrintf("lhs args = %d (should be 5), rhs args = %d (should be 5)\n",nlhs,nrhs); 
        mexErrMsgTxt("Incorrect number of input/output params.");
    }
    
    //input indicies
    const int IDX_GPRIORINV  = 1;
    const int IDX_X  = 2;
    const int IDX_Y  = 3;
    const int IDX_C  = 4;
    const int IDX_B  = 0;
    const int IDX_XB = 5;
    
    if(!mxIsDouble(prhs[IDX_B])) {
        mexErrMsgTxt("input 'filter' must be a floating point array of correct double type!");
    }
    if(!mxIsDouble(prhs[IDX_GPRIORINV])) {
        mexErrMsgTxt("Inverse of the prior covariance must be a floating point array of correct double type!");
    }
    
    int nGPUs;  

    
    long int N = mxGetNumberOfElements(prhs[IDX_B]);
    
    long int N_P = mxGetNumberOfElements(prhs[IDX_GPRIORINV]);
    if(N_P != N*N) {
        mexErrMsgTxt("Inverse of the prior covariance is not the correct size!");
    }
    
    //figures out number of blocks in input
    if( mxIsStruct(prhs[IDX_X]) && mxIsStruct(prhs[IDX_Y]) && mxIsStruct(prhs[IDX_C])) {
        nGPUs = 1;
    }
    else if( (mxIsCell(prhs[IDX_X]) && mxIsCell(prhs[IDX_Y]) && mxIsCell(prhs[IDX_C])) && (mxGetNumberOfElements(prhs[IDX_X]) == mxGetNumberOfElements(prhs[IDX_Y]) && mxGetNumberOfElements(prhs[IDX_X]) == mxGetNumberOfElements(prhs[IDX_C]))) {
        nGPUs = mxGetNumberOfElements(prhs[IDX_Y]);
    }
    else {
        mexErrMsgTxt("Invalid GPU array inputs.\n");
    }
    
    int useXb = 0;
    if(nrhs > IDX_XB && (mxIsCell(prhs[IDX_XB]) || mxIsStruct(prhs[IDX_XB]))) {
        useXb = 1;
    }
    //mexPrintf("useXb = %d\n",useXb); 
    
    //get floating point type used
    int arrayType = -1;
    if( mxIsStruct(prhs[IDX_X]) ) {
        arrayType = kcGetArrayType(prhs[IDX_X]);
    }
    else if( mxIsCell(prhs[IDX_X]) && mxGetNumberOfElements(prhs[IDX_X]) > 0) {
        arrayType = kcGetArrayType(mxGetCell(prhs[IDX_X],0));
    }
    else {
        mexErrMsgTxt("Invalid GPU array inputs.\n");
    }
    
    double * f, *g, *h, *m, *m2, *h_prior;
    
    // Kernels -- 
    //mexPrintf("computing f,g,h\n");
    plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(N,1,mxDOUBLE_CLASS,mxREAL);
    plhs[2] = mxCreateNumericMatrix(N,N,mxDOUBLE_CLASS,mxREAL);
    plhs[3] = mxCreateNumericMatrix(N,N,mxDOUBLE_CLASS,mxREAL);
    plhs[4] = mxCreateNumericMatrix(N,1,mxDOUBLE_CLASS,mxREAL);

    f  = (double*)mxGetPr(plhs[0]);
    g  = (double*)mxGetPr(plhs[1]);
    h  = (double*)mxGetPr(plhs[2]);
    m  = (double*)mxGetPr(plhs[3]);
    m2 = (double*)mxGetPr(plhs[4]);
    h_prior = (double*)mxGetPr(prhs[IDX_GPRIORINV]);
    
    
//     mexPrintf("running on %d plans...\n",nGPUs);
    
    //sets up plans for each block
    const mxArray * Xb;
    if(useXb) {
        Xb = prhs[IDX_XB];
    }
    if(arrayType == KC_FLOAT_ARRAY) {
        runRMMALA<float>(f,g,h,m,m2,h_prior,prhs[IDX_X],prhs[IDX_Y],prhs[IDX_C],prhs[IDX_B],Xb,useXb,nGPUs);
    }
    else if(arrayType == KC_DOUBLE_ARRAY) {
        runRMMALA<double>(f,g,h,m,m2,h_prior,prhs[IDX_X],prhs[IDX_Y],prhs[IDX_C],prhs[IDX_B],Xb,useXb,nGPUs);
    }
    else {
        mexErrMsgTxt("Invalid GPU array inputs.\n");
    }
    
}
