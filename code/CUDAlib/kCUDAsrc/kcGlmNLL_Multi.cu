#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"
#include "reduction.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"
#include "kcMatrixMultHelper.h"


#include <float.h>

template <class T> __device__  T minDenom();
template <> __device__ double minDenom<double>() {
    return 1000*DBL_EPSILON;
}
template <> __device__ float minDenom<float>() {
    return 1000*FLT_EPSILON;
}


//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Structs                       ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T> 
struct GPUGLMplan  {
    //Host-side input data
    long int M; //rows of X
    long int N; //columns of X
    
    // GLM inputs (on GPU)
    T * X;
    T * y;
    T * b;
    
    // compute space (on GPU)
    T * C;
    T * F;  //column of C
    T * G;  //column of C
    T * Xb; //column of C
    
    //answer space
    T * H_ans; //N*(N+1)
    T * G_ans; //N*1

    //answer space
    T * H_ans_host; //N*(N)
    T * G_ans_host; //N*1
    T * F_ans_host; //M
    
    //params
    T dt;
    T * a_p;
    int nParams;
    
    int GLMtype;
    
    unsigned int devNum;
    

    
    //Stream for asynchronous command execution
    cudaStream_t stream;
    
    cublasHandle_t handle;

};

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       computeFGH                    ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T> 
void computeFGH(GPUGLMplan<T> * plan, double * f, double * g, double * h, const long int N, const int nGPUs,  void(*d2llKernel)(T *, T *, T *,const T *, const T *, const T *, const T, const long int, const long int, const T *) ) {
    cublasStatus_t stat;
    
    dim3 block_size;
    block_size.x = 256;
    block_size.y = 1024/block_size.x;
    
    T alpha = (T)1.0;
    T beta  = (T)0.0;
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
                           &beta,
                           plan[ii].Xb, 1);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("initial matrix times filter failed");
        }
        
        
        // configure a two dimensional grid
        dim3 grid_size;
        grid_size.x = plan[ii].M/block_size.x + ((plan[ii].M%block_size.x == 0)? 0:1);
        grid_size.y = plan[ii].N/block_size.y + ((plan[ii].N%block_size.y == 0)? 0:1);

        //mexPrintf("  multiply rows\n");                     //(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N) {
        d2llKernel<<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].C,plan[ii].F,plan[ii].G,plan[ii].X, plan[ii].y, plan[ii].Xb, plan[ii].dt,plan[ii].M, plan[ii].N,plan[ii].a_p);
        getLastCudaError("d2llKernel() execution failed.\n");

        

        //multiplies A'*C, result in H
        //mexPrintf("  X*C \n");
        //cublasSetStream(plan[ii].handle, plan[ii].stream);

        matrixMultiplyHelper(plan[ii].C,plan[ii].C,plan[ii].H_ans,N,N,plan[ii].M,&alpha,&beta,plan[ii].stream,plan[ii].handle);
//         stat =  cublasGEMM(plan[ii].handle,
//                            CUBLAS_OP_T, CUBLAS_OP_N,
//                            N, plan[ii].M, plan[ii].M,
//                            &alpha,
//                            plan[ii].C, plan[ii].M,
//                            plan[ii].C, plan[ii].M,
//                            &beta,
//                            plan[ii].H_ans, N);
        
        /*stat =  cublasDgemv(plan[ii].handle,  CUBLAS_OP_T,
                           plan[ii].M, plan[ii].N,
                           &alpha,
                           plan[ii].X, plan[ii].M,
                           plan[ii].G, 1,
                           &beta,
                           plan[ii].G_ans, 1);*/
        stat =  cublasGEMM(plan[ii].handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           N, 1, plan[ii].M,
                           &alpha,
                           plan[ii].X, plan[ii].M,
                           plan[ii].G, plan[ii].M,
                           &beta,
                           plan[ii].G_ans, N);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("gradient matrix times failed");
        }
        
        
        
        int nb = 0;
        sumIntoBlocks<T>(plan[ii].F,plan[ii].Xb, nb, plan[ii].M,  plan[ii].stream); //in reduction.h
        numBlocks[ii] = nb;
        
        /*cudaStreamSynchronize(plan[ii].stream);
        end = clock();
        time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        mexPrintf("MB_3 %d = %2.4f\n",ii,time_spent);*/
    }        
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        //cublasSetStream(plan[ii].handle, plan[ii].stream);
	    
        checkCudaErrors(cudaMemcpyAsync(plan[ii].H_ans_host, plan[ii].H_ans, N*N *sizeof(T),          cudaMemcpyDeviceToHost, plan[ii].stream));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].G_ans_host, plan[ii].G_ans, N*1 *sizeof(T),          cudaMemcpyDeviceToHost, plan[ii].stream));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].F_ans_host, plan[ii].Xb,     numBlocks[ii] *sizeof(T), cudaMemcpyDeviceToHost, plan[ii].stream));
//         checkCudaErrors(cudaMemcpyAsync(plan[ii].F_ans_host, plan[ii].F,     plan[ii].M *sizeof(T), cudaMemcpyDeviceToHost, plan[ii].stream));
        
        
    }
                
    for(ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("running sum loop plan %d, f = %2.2f\n",ii,f[0]);
        //mexPrintf("  M = %d, N = %d, dt=%2.2f\n", plan[ii].M, plan[ii].N,plan[ii].dt);
        //begin = clock();
        
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cudaStreamSynchronize(plan[ii].stream);
        
        //mexPrintf("  f \n");
        double f_p = 0;
        for(jj = 0; jj <  numBlocks[ii]; jj++) { //plan[ii].M; jj++) {//
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
                h[jj]  = plan[ii].H_ans_host[jj];
            }
            else {
                h[jj] += plan[ii].H_ans_host[jj];
            }
        }
        
        //end = clock();
        //time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        //mexPrintf("FB %d = %2.4f\n",ii,time_spent);
    }
    free(numBlocks);
}
template void computeFGH<float>(GPUGLMplan<float> * plan, double * f, double * g, double * h, const long int N, const int nGPUs,  void(*d2llKernel)(float *, float *, float *,const float *, const float *, const float *, const float, const long int, const long int, const float *));
template void computeFGH<double>(GPUGLMplan<double> * plan, double * f, double * g, double * h, const long int N, const int nGPUs,  void(*d2llKernel)(double *, double *, double *,const double *, const double *, const double *, const double, const long int, const long int, const double *));



//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       computeFG                     ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T> 
void computeFG(GPUGLMplan<T> * plan, double * f, double * g, const long int N, const int nGPUs, void(*dllKernel)(T *, T *, const T *, const T *, const T, const long int,const T *) ) {
    //clock_t begin,begin_0, end;
    //double time_spent;
    //begin_0 = clock();
    
    cublasStatus_t stat;
    
    int block_size = 1024;
    
    T alpha = (T)1.0;
    T beta  = (T)0.0;
    long int ii,jj;
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
                           &beta,
                           plan[ii].Xb, 1);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("initial matrix times filter failed");
        }
        /*cudaStreamSynchronize(plan[ii].stream);
        end = clock();
        time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        mexPrintf("MV 1 %d = %2.4f\n",ii,time_spent);*/
        
        // configure a one dimensional grid
        int grid_size = plan[ii].M/block_size + ((plan[ii].M%block_size == 0)? 0:1);

        
        //begin = clock();
        
        dllKernel<<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].F,plan[ii].G,plan[ii].Xb,plan[ii].y,plan[ii].dt,plan[ii].M,plan[ii].a_p);
        getLastCudaError("dllKernel() execution failed.\n");
        
        /*cudaStreamSynchronize(plan[ii].stream);
        end = clock();
        time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        mexPrintf("DLL %d = %2.4f\n",ii,time_spent);*/
        
        
        /*begin = clock();
        stat =  cublasDgemv(plan[ii].handle,  CUBLAS_OP_T,
                               plan[ii].M, plan[ii].N,
                               &alpha,
                               plan[ii].X, plan[ii].M,
                               plan[ii].G, 1,
                               &beta,
                               plan[ii].G_ans, 1);
        cudaStreamSynchronize(plan[ii].stream);
        end = clock();
        time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        mexPrintf("MV %d = %2.4f\n",ii,time_spent);*/
        
        //begin = clock();
        //GEMM seems faster for this shape of matrix vector multiplication than GEMV
        stat =  cublasGEMM(plan[ii].handle,
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   N, 1, plan[ii].M,
                                   &alpha,
                                   plan[ii].X, plan[ii].M,
                                   plan[ii].G, plan[ii].M,
                                   &beta,
                                   plan[ii].G_ans, N);
        
        /*cudaStreamSynchronize(plan[ii].stream);
        end = clock();
        time_spent = ((double)(end - begin))/((double)CLOCKS_PER_SEC);
        mexPrintf("MM 2 %d = %2.4f\n",ii,time_spent);*/
        
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("matrix multiplication failed");
        }
        
        int nb = 0;
        sumIntoBlocks<T>(plan[ii].F,plan[ii].Xb, nb, plan[ii]. M,  plan[ii].stream);
        numBlocks[ii] = nb;
    }
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].F_ans_host, plan[ii].Xb,    numBlocks[ii] *sizeof(T), cudaMemcpyDeviceToHost, plan[ii].stream));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].G_ans_host, plan[ii].G_ans, N*1 *sizeof(T),          cudaMemcpyDeviceToHost, plan[ii].stream));
    }
    
                
    for(ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("running sum loop plan %d, f = %2.2f\n",ii,f[0]);
        //mexPrintf("  M = %d, N = %d, dt=%2.2f\n", plan[ii].M, plan[ii].N,plan[ii].dt);
        
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
        
        
    }
    free(numBlocks);
    
    
    //end = clock();
    //time_spent = ((double)(end - begin_0))/((double)CLOCKS_PER_SEC);
    //mexPrintf("totalTime %d = %2.4f\n",ii,time_spent);
}
template void computeFG<float> (GPUGLMplan<float> * plan, double * f, double * g, const long int N, const int nGPUs, void(*dllKernel)(float*, float *, const float *, const float *, const float, const long int,const float *) );
template void computeFG<double>(GPUGLMplan<double> * plan, double * f, double * g, const long int N, const int nGPUs, void(*dllKernel)(double *, double *, const double *, const double *, const double, const long int,const double *) );

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       computeF                      ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T> 
void computeF(GPUGLMplan<T> * plan, double * f, const long int N, const int nGPUs, void(*llKernel)(T *, const T *,  const T *, const T, const long int, const T *) ) {
    cublasStatus_t stat;
    
    dim3 block_size;
    block_size.x = 1024;
    block_size.y = 1;
    
    T alpha = (T)1.0;
    T beta  = (T)0.0;
    long int ii,jj;
    
    int * numBlocks = (int *)malloc(nGPUs*sizeof(int));
    
    for(ii = 0; ii < nGPUs; ii++) {
        
        //mexPrintf("running comp loop plan %d\n",ii);
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        cublasSetStream(plan[ii].handle, plan[ii].stream);
        
        //mexPrintf("  X*b \n");
        stat =  cublasGEMV(plan[ii].handle,  CUBLAS_OP_N,
                           plan[ii].M, plan[ii].N,
                           &alpha,
                           plan[ii].X, plan[ii].M,
                           plan[ii].b, 1,
                           &beta,
                           plan[ii].Xb, 1);
        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("initial matrix times filter failed");
        }
        
        // configure a two dimensional grid
        dim3 grid_size;
        grid_size.x = plan[ii].M/block_size.x + ((plan[ii].M%block_size.x == 0)? 0:1);
        grid_size.y = 1;//plan[ii].N/block_size.y + ((plan[ii].N%block_size.y == 0)? 0:1);

        llKernel<<<grid_size,block_size,0,plan[ii].stream>>>(plan[ii].F,plan[ii].Xb,plan[ii].y,plan[ii].dt,plan[ii].M,plan[ii].a_p);
        getLastCudaError("llKernel() execution failed.\n");
        
        int nb = 0;
        sumIntoBlocks<T>(plan[ii].F,plan[ii].Xb, nb, plan[ii].M,  plan[ii].stream);
        numBlocks[ii] = nb;
        //cudaStreamSynchronize(plan[ii].stream);
    }
    
    for(ii = 0; ii < nGPUs; ii++) {
        checkCudaErrors(cudaSetDevice(plan[ii].devNum));
        checkCudaErrors(cudaMemcpyAsync(plan[ii].F_ans_host, plan[ii].Xb,     numBlocks[ii]*sizeof(T), cudaMemcpyDeviceToHost, plan[ii].stream));
    }
    
                
    for(ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("running sum loop plan %d, f = %2.2f\n",ii,f[0]);
        //mexPrintf("  M = %d, N = %d, dt=%2.2f\n", plan[ii].M, plan[ii].N,plan[ii].dt);
        
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
    }
    free(numBlocks);
}
template void computeF<float> (GPUGLMplan<float> * plan, double * f, const long int N, const int nGPUs, void(*llKernel)(float *, const float *,  const float *, const float, const long int, const float *) );
template void computeF<double>(GPUGLMplan<double> * plan, double * f, const long int N, const int nGPUs, void(*llKernel)(double *, const double *,  const double *, const double, const long int, const double *) );

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Setup Plan                    ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T>
void setupPlan(GPUGLMplan<T> &plan, const mxArray * X, const mxArray * y, const mxArray * C, const mxArray * b, double dt, const int computeHessian, const int nParams,const double * extraParams, const int GLMtype) {

    plan.devNum = kcGetArrayDev(X);
    
    if(kcGetArrayDev(y) != plan.devNum || kcGetArrayDev(C) != plan.devNum) {
        mexErrMsgTxt("Arrays not all placed on proper devices!\n");
    }
    
    checkCudaErrors(cudaSetDevice(plan.devNum));
    checkCudaErrors(cudaStreamCreate(&(plan.stream)));
    
    
    plan.M = kcGetArraySize(X, 0);
    plan.N = kcGetArraySize(X, 1);
    
    
    int columnsForHessian;
    if(computeHessian) {
        columnsForHessian = plan.N;
    }
    else {
        columnsForHessian = 0;
    }
    
    //checks input sizes
    long int N_C = kcGetArraySize(C, 1);
    long int M_C = kcGetArraySize(C, 0);
    if(plan.M * (columnsForHessian + 3) > N_C*M_C) {
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
    
    plan.nParams = nParams;
    plan.GLMtype = GLMtype;
    T * ep_f;
    int freeEpf = 0;
    if(plan.nParams > 0) {
        int nParamsTotal = plan.nParams;

        checkCudaErrors(cudaMalloc((void**)&(plan.a_p),sizeof(T)*nParamsTotal));
        
        ep_f = (T*)malloc(nParamsTotal*sizeof(T));
        for(int ii = 0; ii < nParamsTotal; ii++) {
            ep_f[ii] = (T)(extraParams[ii]);
        }
        freeEpf = 1;
        checkCudaErrors(cudaMemcpyAsync((plan.a_p),ep_f,  sizeof(T)*nParamsTotal,cudaMemcpyHostToDevice,plan.stream));
        
    }
    
    plan.dt = (T)dt;
    //mexPrintf("plan.dt = %.4f\n",plan.dt);
    
    
    plan.G  = plan.C+plan.M*(columnsForHessian+0);
    plan.F  = plan.C+plan.M*(columnsForHessian+1);
    plan.Xb = plan.C+plan.M*(columnsForHessian+2);
    
    plan.H_ans_host = (T*)malloc(plan.N*plan.N*sizeof(T));
    plan.G_ans_host = (T*)malloc(plan.N*sizeof(T));
    plan.F_ans_host = (T*)malloc(plan.M*sizeof(T));
            
    
    checkCudaErrors(cudaMalloc((void**)&(plan.b),sizeof(T)*plan.N));
    
    
    T * b_f = (T*)malloc(plan.N*sizeof(T));
    double *   b_p = (double *)mxGetPr(b);
    for(int ii = 0; ii < plan.N; ii++) {
        b_f[ii] = (T)(b_p[ii]);
    }
    checkCudaErrors(cudaMemcpyAsync(plan.b,b_f,  sizeof(T)*plan.N,cudaMemcpyHostToDevice,plan.stream));
    
    checkCudaErrors(cudaMalloc((void**)&(plan.H_ans),sizeof(T)*(plan.N+1)*plan.N));
    plan.G_ans = plan.H_ans+plan.N*(columnsForHessian+0);
    
    cublasStatus_t stat;
    stat = cublasCreate(&(plan.handle));
    if(stat != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS initialization failed\n");
    }
    cudaStreamSynchronize(plan.stream);
    free(b_f);
    if(freeEpf) {
        free(ep_f);
    }
}
template void setupPlan<float>(GPUGLMplan<float> &plan, const mxArray * X, const mxArray * y, const mxArray * C, const mxArray * b, double dt, const int computeHessian, const int nParams,const double * extraParams, const int GLMtype);
template void setupPlan<double>(GPUGLMplan<double> &plan, const mxArray * X, const mxArray * y, const mxArray * C, const mxArray * b, double dt, const int computeHessian, const int nParams,const double * extraParams, const int GLMtype);


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
    checkCudaErrors(cudaFree(plan.b));
    checkCudaErrors(cudaFree(plan.H_ans));
    if(plan.nParams > 0) {
        checkCudaErrors(cudaFree(plan.a_p));
    }
    
    free(plan.H_ans_host);
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
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

//__global__ void llKernel(  double * f, const double * Xb,  const double * y, const double dt, const long int M,double * a_p)
//__global__ void dllKernel( double * f, double * g, double * Xb, const double * y,  const double dt, const long int M,double * a_p)
//__global__ void d2llKernel(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N, double * a_p)



/*functions to compute log likelihood and derivatives for one observation in GLM
* ll   computes only function
* dll  computes function, and gradient! 
* d2ll computes function, gradient, and hessian!
* duplication of functionality exists for optimization
*
* inputs:
*   M   = number of observations (num rows of design matrix)
*   N   = number of coefficients (num columns of design matrix)
*   dt  = observation bin size (in seconds)
*   Xb  = X*b where X is the design matrix and b is the weight vector (vector of M terms)
*   y   = observed spike counts in each bin (vector of M terms)
*   a_p = vector of any extra parameters needed to define link function or whatever
*
* outputs
*  f    = vector of length M for the log likelihood of each observation
*  g    = vector of length M for computing the gradient of the likelihood of each observation
*  C    = matrix of size MxN for computing the hessian of the likelihood
*   
*
* log likelihood of one observation is GLM is \log p(y[ii] | Xb[ii], dt)
*   where Xb = X[ii,:]*b and Xb[ii,:] is the ii'th row of the design matrix
*   and b is the weight vector
*
* let ii = blockIdx.x*blockDim.x+threadIdx.x; 
*     jj = blockIdx.y*blockDim.y+threadIdx.y; 
*
* \log p(y[ii] | Xb[ii], dt) = f[ii]
* \frac{d}{db}     \log p(y[ii] | X[ii],b[ii],dt)     = g[ii]*X[ii]   //only compute the coefficients on the X's here, X is multiplied in after
* \frac{d^2}{db_jj db} \log p(y[ii] | X[ii],b[ii],dt) = C[ii,jj]*X[ii]   //only compute the M by N matrix here, X' is multiplied in after to give the Hessian
*            in matlab, C would be bsxfun(@times,X,hh) for some vector hh of coefficients
*
*/

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Poisson                  ==============================================
//==========================================                    Link  : Exp                      ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T>
__global__ void llKernel_ExpPoisson(T * f, const T * Xb,  const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T eXb = safeExp(Xb[row] + log(dt));
        f[row] = -y[row]*Xb[row] + eXb;
    }
}
template __global__ void llKernel_ExpPoisson<float>(float * f, const float * Xb,  const float * y, const float dt, const long int M, const float * a_p);
template __global__ void llKernel_ExpPoisson<double>(double * f, const double * Xb,  const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void dllKernel_ExpPoisson(T * f, T * g, const T * Xb, const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cXb = Xb[row];
        T cy  = y[row];
        T ceXb = safeExp(cXb + log(dt));
        
        f[row]   = -cy*cXb + ceXb;
        g[row]   = ceXb - cy;
    }
}
template __global__ void dllKernel_ExpPoisson<float>(float * f, float * g, const float * Xb, const float * y, const float dt, const long int M, const float * a_p);
template __global__ void dllKernel_ExpPoisson<double>(double * f, double * g, const double * Xb, const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void d2llKernel_ExpPoisson(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T dt, const long int M, const long int N, const T * a_p) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T cXb  = Xb[row];
            T ldt = log(dt);
            T ceXb_12 = (cXb + ldt)/T(2);
            
            C[row + M*col] = X[row + M*col] *safeExp(ceXb_12);
            
            if(col == 0) {
                T ceXb = safeExp(cXb+ldt);
                //double ceXb_n12 = exp(min(KC_EXP_MAX,0.5*(cXb + log(dt))));
                T cy  = y[row];
                f[row]   = -cy*cXb      + ceXb;
                g[row]   = -cy          + ceXb;
            }
        }
    }
}
template __global__ void d2llKernel_ExpPoisson<float>(float * C, float * f, float * g,const float * X, const float * y, const float * Xb, const float dt, const long int M, const long int N, const float * a_p);
template __global__ void d2llKernel_ExpPoisson<double>(double * C, double * f, double * g,const double* X, const double * y, const double * Xb, const double dt, const long int M, const long int N, const double * a_p);


//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Poisson                  ==============================================
//==========================================                    Link  : Linear                   ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
__global__ void llKernel_LinPoisson(T * f, const T * Xb,  const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cy  = y[row];
        T cXb  = Xb[row];
        T ceXb = exp(cXb);
        T r = Xb[row]<35?log1p(ceXb):Xb[row];

        f[row]   = -cy*log(r) + dt*r;
    }
}
template __global__ void llKernel_LinPoisson<float>(float * f, const float * Xb,  const float * y, const float dt, const long int M, const float * a_p);
template __global__ void llKernel_LinPoisson<double>(double * f, const double * Xb,  const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void dllKernel_LinPoisson(T * f, T * g, const T * Xb, const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cy  = y[row];
        T cXb  = Xb[row];
        T ceXb = exp(cXb);
        T cneXb = exp(-cXb);
        T inv_dr = (T(1)+cneXb);
        T r = Xb[row]<35?log1p(ceXb):Xb[row];


        T inv_dr_r;
        if(Xb[row] > 35) {
            inv_dr_r = r;
        }
        else if(Xb[row]  < -35) {
            inv_dr_r = 1;
        }
        else {
            inv_dr_r = r*(T(1)+cneXb);
        }


        f[row]   = -cy*log(r) + dt*r;
        //g[row]   = rp*(dt - cy/max(KC_POS_DENOM_MIN,lexp));
        g[row]   = (dt/inv_dr - cy/inv_dr_r);
    }
}
template __global__ void dllKernel_LinPoisson<float>(float * f, float * g, const float * Xb, const float * y, const float dt, const long int M, const float * a_p);
template __global__ void dllKernel_LinPoisson<double>(double * f, double * g, const double * Xb, const double * y, const double dt, const long int M, const double * a_p);


template <class T>
__global__ void d2llKernel_LinPoisson(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T dt, const long int M, const long int N, const T * a_p) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T cy  = y[row];
            T cXb  = Xb[row];
            T ceXb = exp(cXb);
            T cneXb = exp(-cXb);
            T inv_dr = (T(1)+cneXb);
            T r = Xb[row]<35?log1p(ceXb):Xb[row];
            
            
            T inv_dr_r;
            if(Xb[row] > 35) {
                inv_dr_r = r;
            }
            else if(Xb[row]  < -35) {
                inv_dr_r = 1;
            }
            else {
                inv_dr_r = r*(T(1)+cneXb);
            }

            
            
            T inv_d2r   = inv_dr*(T(1)+ceXb);
            T inv_d2r_r  = inv_dr_r*(T(1)+ceXb);
            
            
            T m = sqrt(max(T(0),dt/inv_d2r+ cy/(inv_dr_r*inv_dr_r) - cy/inv_d2r_r));
            C[row + M*col] = X[row + M*col] *m;
            
            if(col == 0) {
                
                f[row]   = -cy*log(r) + dt*r;
                //g[row]   = rp*(dt - cy/max(KC_POS_DENOM_MIN,lexp));
                g[row]   = (dt/inv_dr - cy/inv_dr_r);
            }
        }
    }
}
template __global__ void d2llKernel_LinPoisson<float>(float * C, float * f, float * g,const float * X, const float * y, const float * Xb, const float dt, const long int M, const long int N, const float * a_p);
template __global__ void d2llKernel_LinPoisson<double>(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N, const double * a_p);

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Poisson                  ==============================================
//==========================================                    Link  : Power-Law                ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
__global__ void llKernel_PowPoisson(T * f, const T * Xb,  const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T p = a_p[0];
        T cy  = y[row];
        T ceXb = exp(Xb[row]);
        T a = Xb[row]<35?log1p(ceXb):Xb[row];
        T r = pow(a,p);
                
        f[row]   = -cy*p*log(a) + dt*r;
    }
}
template __global__ void llKernel_PowPoisson<float>(float * f, const float * Xb,  const float * y, const float dt, const long int M, const float * a_p);
template __global__ void llKernel_PowPoisson<double>(double * f, const double * Xb,  const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void dllKernel_PowPoisson(T * f, T * g, const T * Xb, const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T p = a_p[0];
        T cy  = y[row];
        T ceXb = exp(Xb[row]);
        T cneXb = exp(-Xb[row]);
        T a = Xb[row]<35?log1p(ceXb):Xb[row];
        T r = pow(a,p);

        T inv_dr = (1+cneXb)*pow(a,1-p)/p; //inv_dr
        T inv_dr_r; //inv_dr_r
        if(Xb[row]  < -35) {
            //inv_dr = T(1)/p;
            inv_dr_r = T(1)/p;
        }
        else if(Xb[row]  < 35 ) {
            //inv_dr = pow(a,p-1)/p;
            inv_dr_r = a*(1+cneXb)/p;
        }
        else {
            //inv_dr = pow(a,p-1)*(1+cneXb)/p;
            inv_dr_r = Xb[row]/p;
        }




        f[row]   = -cy*p*log(a) + dt*r;

        g[row]   = dt/inv_dr - cy/inv_dr_r;
    }
}
template __global__ void dllKernel_PowPoisson<float>(float * f, float * g, const float * Xb, const float * y, const float dt, const long int M, const float * a_p);
template __global__ void dllKernel_PowPoisson<double>(double * f, double * g, const double * Xb, const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void d2llKernel_PowPoisson(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T dt, const long int M, const long int N, const T * a_p) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T p = a_p[0];
            T cy  = y[row];
            T ceXb = exp(Xb[row]);
            T cneXb = exp(-Xb[row]);
            T a = Xb[row]<35?log1p(ceXb):Xb[row];
            T r = pow(a,p);
            
            T inv_dr = (1+cneXb)*pow(a,1-p)/p; //inv_dr
            T inv_dr_r; //inv_dr_r
            if(Xb[row]  < -35) {
                //inv_dr = T(1)/p;
                inv_dr_r = T(1)/p;
            }
            else if(Xb[row]  < 35 ) {
                //inv_dr = pow(a,p-1)/p;
                inv_dr_r = a*(1+cneXb)/p;
            }
            else {
                //inv_dr = pow(a,p-1)*(1+cneXb)/p;
                inv_dr_r = Xb[row]/p;
            }
            
            T h = T(1)/(1+ceXb) + (p-1)/(p*inv_dr_r);
            //double d2r = inv_dr/h;
            //double d2r_r = inv_dr_r/h;
            T m = sqrt(max(T(0),cy/(inv_dr_r*inv_dr_r) + (dt*h/inv_dr - cy*h/inv_dr_r)));
                
            C[row + M*col] = X[row + M*col] *m;
            
            if(col == 0) {
                
                
                
                f[row]   = -cy*p*log(a) + dt*r;

                g[row]   = dt/inv_dr - cy/inv_dr_r;
            }
        }
    }
}
template __global__ void d2llKernel_PowPoisson<float>(float * C,float * f, float * g,const float * X, const float * y, const float * Xb, const float dt, const long int M, const long int N, const float * a_p);
template __global__ void d2llKernel_PowPoisson<double>(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N, const double * a_p);



//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Bernoulli                ==============================================
//==========================================                    Link  : Power-Law                ==============================================
//=============================================================================================================================================
//=============================================================================================================================================
template <class T>
__global__ void llKernel_PowBer(T * f, const T * Xb,  const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T a = a_p[0];
        T cy  = y[row];
                
        T l;  //ex
        if(Xb[row] > 35) {
            l = Xb[row];
        }
        else {
            l = log1p(exp(Xb[row]));
        }
        T r = pow(l,a)*dt; //r
        
        //f = -a
        if(cy > 0) {
            if(r < 1) {
                f[row] = -log(-expm1(-r));
            }
            else {
                f[row] = -log1p(-exp(-r ));
            }
        }
        else {
            f[row] = r;
        }
    }
}
template __global__ void llKernel_PowBer<float>(float * f, const float * Xb,  const float * y, const float dt, const long int M, const float * a_p);
template __global__ void llKernel_PowBer<double>(double * f, const double * Xb,  const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void dllKernel_PowBer(T * f, T * g, const T * Xb, const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T p = a_p[0];
        

        T cy  = y[row];
                
        T l;  //ex
        if(Xb[row] > 35) {
            l = Xb[row];
        }
        else {
            l = log1p(exp(Xb[row]));
        }
        T r = pow(l,p)*dt; //r
            
        T nex = T(1)/(1+exp(-Xb[row])); //nex
        T inv_dr = pow(l,p-1); //inv_dr
        
        //f = -a
        if(cy > 0) {
            if(r < 1) {
                f[row] = -log(-expm1(-r));
            }
            else {
                f[row] = -log1p(-exp(-r ));
            }
            T exprm1 = expm1(min(T(200),r));
            g[row] = (exprm1 > minDenom<T>()) ? -((dt*p)*inv_dr*(nex/exprm1)) : -(p*(1 - nex));
        }
        else {
            f[row] = r;
            g[row] = (dt*nex*p*inv_dr);
        }
    }
}
template __global__ void dllKernel_PowBer<float>(float * f, float * g, const float * Xb, const float * y, const float dt, const long int M, const float * a_p);
template __global__ void dllKernel_PowBer<double>(double * f, double * g, const double * Xb, const double * y, const double dt, const long int M, const double * a_p);

template <class T>
__global__ void d2llKernel_PowBer(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T dt, const long int M, const long int N, const T * a_p) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T p = a_p[0];
        

            T cy  = y[row];

            T l;  //ex
            if(Xb[row] > 35) {
                l = Xb[row];
            }
            else {
                l = log1p(exp(Xb[row]));
            }
            T r = pow(l,p)*dt; //r

            T nex = T(1)/(1+exp(-Xb[row])); //nex
            T inv_dr = pow(l,p-1); //inv_dr

            //f = -a
            T dll;
            T ll;
            T m;
            if(cy > 0) {
                ll = (r<1) ? (-log(-expm1(-r))):(-log1p(-exp(-r )));
                T exprm1 = expm1(r);
                dll = (exprm1 > minDenom<T>()) ? ((dt*p)*inv_dr*(nex/exprm1)) : (p*(1 - nex));
                
                m = -(dll*((1-nex) + (p-1)/l) - dll*dll*exp(min(T(200),r)));
            }
            else {
                ll = r;
                dll = (-dt*nex*p*inv_dr);
                m = -dll*((1-nex) + (p-1)/l);
            }
            
            
            
            C[row + M*col] = X[row + M*col] *sqrt(max(T(0),m));
            if(col == 0) {
                
                
                f[row] = ll;
                g[row] = -dll;
                
            }
            
        }
    }
}
template __global__ void d2llKernel_PowBer<float>(float * C,float * f, float * g,const float * X, const float * y, const float * Xb, const float dt, const long int M, const long int N, const float * a_p);
template __global__ void d2llKernel_PowBer<double>(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N, const double * a_p);



//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Bernoulli                ==============================================
//==========================================                    Link  : Exp                      ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
__global__ void d2llKernel_ExpBer(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T dt, const long int M, const long int N, const T * a_p) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T cy  = y[row];
            T Xw  = Xb[row] + log(dt);
            T r = exp(min(T(200),Xw));
            T a = (cy > 0) ? ( (r < 1) ? log(-expm1(-r)) : log1p(-exp(-r))) : -r; 
            T b = (cy > 0) ? (exp(min(T(200),-r - a + Xw))) : -r;
            T c = (cy > 0) ? (b*(-r - b + 1)) : -r;
            
            C[row + M*col] = X[row + M*col] *sqrt(max(T(0),-c));
            if(col == 0) {
                
                
                f[row] = -a;
                g[row] = -b;
                
            }
        }
    }
}
template __global__ void d2llKernel_ExpBer<float>(float * C, float * f, float * g,const float * X, const float * y, const float * Xb, const float dt, const long int M, const long int N, const float * a_p);
template __global__ void d2llKernel_ExpBer<double>(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt, const long int M, const long int N, const double * a_p);

template <class T>
__global__ void dllKernel_ExpBer(T * f, T * g, const T * Xb, const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cy  = y[row];
        T Xw  = Xb[row] + log(dt);
        T r = exp(min(T(200),Xw));
        T a = (cy > 0) ? ( (r < 1) ? log(-expm1(-r)) : log1p(-exp(-r))) : -r; 
        T b = (cy > 0) ? (exp(min(T(200),-r - a + Xw))) : -r;
        f[row] = -a;
        g[row] = -b;
    }
}
template __global__ void dllKernel_ExpBer<float>(float * f, float * g, const float * Xb, const float * y, const float dt, const long int M, const float * a_p);
template __global__ void dllKernel_ExpBer<double>(double * f, double * g, const double * Xb, const double * y, const double dt, const long int M, const double * a_p);


template <class T>
__global__ void llKernel_ExpBer(T * f, const T * Xb,  const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cy  = y[row];
        T Xw  = Xb[row] + log(dt);
        T r = exp(min(T(200),Xw));
        T a = (cy > 0) ? ( (r < 1) ? log(-expm1(-r)) : log1p(-exp(-r))) : -r; 

        f[row] = -a;
    }
}
template __global__ void llKernel_ExpBer<float>(float * f, const float * Xb,  const float * y, const float dt, const long int M, const float * a_p);
template __global__ void llKernel_ExpBer<double>(double * f, const double * Xb,  const double * y, const double dt, const long int M, const double * a_p);

//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                 Kernels                             ==============================================
//==========================================                    Approx: Bernoulli                ==============================================
//==========================================                    Link  : Softrec                  ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
__global__ void d2llKernel_LinBer(T * C, T * f, T * g,const T * X, const T * y, const T * Xb, const T dt, const long int M, const long int N, const T * a_p) {
    long int row = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int col = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(row < M) {
        if(col < N) {
            T cy  = y[row];
            T ex = exp(min(T(200),Xb[row]));
            T r = (Xb[row]<35)?log1p(ex)*dt:Xb[row]*dt;
            T nex = T(1)/(1+exp(min(T(200),-Xb[row])));
            
            T a = (cy > 0) ? ( (r < 1) ? log(-expm1(-r)) : log1p(-exp(-r))) : -r; 
            T b;
            T c;
            if(cy > 0) {
                T expr = (r>0.01)? pow((1+ex),dt)-1 : expm1(r);
                b = (expr > minDenom<T>()) ? (dt*nex/expr) : dt*(1-nex);
                c = (expr > minDenom<T>()) ? (dt*nex*(1-nex)/(expr)  - b*b*(expr + 1)) : (dt*(1-nex)*(1-nex)  - b*b*(expr+1));
            }
            else {
                b = -dt*nex;
                c = -dt*nex*(1-nex);
            }
            
            C[row + M*col] = X[row + M*col] *sqrt(max(T(0),-c));
            if(col == 0) {
                
                
                f[row] = -a;
                g[row] = -b;
                
            }
        }
    }
}
template __global__ void d2llKernel_LinBer<float>(float * C, float * f, float * g,const float * X, const float * y, const float * Xb, const float dt,  const long int M, const long int N, const float * a_p);
template __global__ void d2llKernel_LinBer<double>(double * C, double * f, double * g,const double * X, const double * y, const double * Xb, const double dt,  const long int M, const long int N, const double * a_p);

template <class T>
__global__ void dllKernel_LinBer(T * f, T * g, const T * Xb, const T * y, const T dt,  const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cy  = y[row];
        T ex = exp(min(T(200),Xb[row]));
        T r = (Xb[row]<35)?log1p(ex)*dt:Xb[row]*dt;
        T nex = T(1)/(1+exp(min(T(200),-Xb[row])));

        T a = (cy > 0) ? ( (r < 1) ? log(-expm1(-r)) : log1p(-exp(-r))) : -r; 
        T b;
        if(cy > 0) {
            T expr = (r>0.01)? pow((1+ex),dt)-1 : expm1(r);
            b = (expr > minDenom<T>()) ? (dt*nex/expr) : dt*(1-nex);
        }
        else {
            b = -dt*nex;
        }


        f[row] = -a;
        g[row] = -b;
    }
}
template __global__ void dllKernel_LinBer<float>(float * f, float * g, const float * Xb, const float * y, const float dt,  const long int M, const float * a_p);
template __global__ void dllKernel_LinBer<double>(double * f, double * g, const double * Xb, const double * y, const double dt,  const long int M, const double * a_p);

template <class T>
__global__ void llKernel_LinBer(T * f, const T * Xb,  const T * y, const T dt, const long int M, const T * a_p) {
    long int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < M) {
        T cy  = y[row];
        T ex = exp(min(T(200),Xb[row]));
        T r = (Xb[row]<35)?log1p(ex)*dt:Xb[row]*dt;
        T nex = T(1)/(1+exp(min(T(200),-Xb[row])));

        T a = (cy > 0) ? ( (r < 1) ? log(-expm1(-r)) : log1p(-exp(-r))) : -r; 

        f[row] = -a;
    }
}
template __global__ void llKernel_LinBer<float>(float * f, const float * Xb,  const float * y, const float dt, const long int M, const float * a_p);
template __global__ void llKernel_LinBer<double>(double * f, const double * Xb,  const double * y, const double dt, const long int M, const double * a_p);




//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Main                          ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
void computeNLL(double * f, double * g, double * h,const mxArray* GPU_X, const mxArray* GPU_Y, const mxArray* GPU_C, const mxArray* b, const int N, const double dt, const int GLMtype, const double * extraParams, const int nExtraParams, const int computeLevel) {
    int computeHessian = (computeLevel > 2);
    

    
    //figures out number of blocks in input
    int nGPUs;
    if( mxIsStruct(GPU_X) && mxIsStruct(GPU_Y) && mxIsStruct(GPU_C)) {
        nGPUs = 1;
    }
    else if( (mxIsCell(GPU_X) && mxIsCell(GPU_Y) && mxIsCell(GPU_C)) && (mxGetNumberOfElements(GPU_X) == mxGetNumberOfElements(GPU_Y) && mxGetNumberOfElements(GPU_X) == mxGetNumberOfElements(GPU_C))) {
        nGPUs = mxGetNumberOfElements(GPU_X);
    }
    else {
        mexErrMsgTxt("Invalid GPU array inputs.\n");
    }
    //mexPrintf("running on %d GPUs. dt = %2.2f\n",nGPUs,dt);
    
    //sets up plans for each block
    GPUGLMplan<T> plans[nGPUs];
    if( mxIsStruct(GPU_X)) {
        setupPlan<T>( plans[0], GPU_X, GPU_Y, GPU_C, b,dt,computeHessian,nExtraParams,extraParams,GLMtype);
    }
    else {
        for(int ii = 0; ii < nGPUs; ii++) {
            setupPlan<T>( plans[ii], mxGetCell(GPU_X,ii), mxGetCell(GPU_Y,ii), mxGetCell(GPU_C,ii), b,dt,computeHessian,nExtraParams,extraParams,GLMtype);
        }
    }
    
    //mexPrintf("computing N=%d\n",N);
    
    
    // Kernels -- 
    void (*HKernel)(T *, T *, T *,const T *, const T *, const T *, const T, const long int, const long int, const T *);
    void(*GKernel)(T *, T *, const T *, const T *, const T, const long int, const T *);
    void(*FKernel)(T *, const T *,  const T *, const T, const long int, const T *) ;
    int nExtraParamsRequired = 0;
    //=======================================================================
    // add new functions for changing link function
    switch(GLMtype) {
        case 5:
            HKernel = &d2llKernel_PowPoisson<T>;
            GKernel = &dllKernel_PowPoisson<T>;
            FKernel = &llKernel_PowPoisson<T>;
            nExtraParamsRequired = 1;
            break;
        case 4:
            HKernel = &d2llKernel_LinPoisson<T>;
            GKernel = &dllKernel_LinPoisson<T>;
            FKernel = &llKernel_LinPoisson<T>;
            nExtraParamsRequired = 0;
            break;
        case 3:
            HKernel = &d2llKernel_PowBer<T>;
            GKernel = &dllKernel_PowBer<T>;
            FKernel = &llKernel_PowBer<T>;
            nExtraParamsRequired = 1;
            break;
        case 2:
            HKernel = &d2llKernel_LinBer<T>;
            GKernel = &dllKernel_LinBer<T>;
            FKernel = &llKernel_LinBer<T>;
            nExtraParamsRequired = 0;
            break;
        case 1:
            HKernel = &d2llKernel_ExpBer<T>;
            GKernel = &dllKernel_ExpBer<T>;
            FKernel = &llKernel_ExpBer<T>;
            nExtraParamsRequired = 0;
            break;
        case 0:
            HKernel = &d2llKernel_ExpPoisson<T>;
            GKernel = &dllKernel_ExpPoisson<T>;
            FKernel = &llKernel_ExpPoisson<T>;
            nExtraParamsRequired = 0;
            break;
        default:
            HKernel = &d2llKernel_ExpPoisson<T>;
            GKernel = &dllKernel_ExpPoisson<T>;
            FKernel = &llKernel_ExpPoisson<T>;
            nExtraParamsRequired = 0;
            break;
    }
    //=======================================================================
    if(nExtraParams < nExtraParamsRequired) {
        mexErrMsgTxt("Insufficent parameters for GLM type.\n");
    }
    if(GLMtype == 3 && extraParams[0] == 1.0) {
        HKernel = &d2llKernel_LinBer<T>;
        GKernel = &dllKernel_LinBer<T>;
        FKernel = &llKernel_LinBer<T>;
        nExtraParamsRequired = 0;
    }
    if(GLMtype == 5 && extraParams[0] == 1.0) {
        HKernel = &d2llKernel_LinPoisson<T>;
        GKernel = &dllKernel_LinPoisson<T>;
        FKernel = &llKernel_LinPoisson<T>;
        nExtraParamsRequired = 0;
    }
    
    
    
    if(computeLevel == 3) {
        //mexPrintf("computing f,g,h\n");
        
       
        computeFGH<T>(plans, f, g, h, N,  nGPUs, HKernel);
        
    }
    else if(computeLevel == 2) {
        //mexPrintf("computing f,g\n");


       
        computeFG<T>(plans, f, g, N,  nGPUs, GKernel);
        
    }
    else if(computeLevel == 1) {
        //mexPrintf("computing f\n");
       
        computeF<T>(plans, f, N,  nGPUs, FKernel);
        //(GPUGLMplan<float> [nGPUs], float *, const int, int, void (*)(float *, const float *, const float *, float, long, const float *))
    }
    
    for(int ii = 0; ii < nGPUs; ii++) {
        //mexPrintf("destroying GPU plan %d\n",ii);
        destroyPlan( plans[ii]);
    }
}
template void computeNLL<float>(double * f, double * g, double * h,const mxArray * GPU_X, const mxArray * GPU_Y, const mxArray* GPU_C, const mxArray*  b, const int N, const double dt, const int GLMtype, const double * extraParams, const int nExtraParams, const int computeLevel);
template void computeNLL<double>(double * f, double * g, double * h,const mxArray * GPU_X, const mxArray * GPU_Y, const mxArray* GPU_C, const mxArray* b, const int N, const double dt, const int GLMtype, const double * extraParams, const int nExtraParams, const int computeLevel);


//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       Main                          ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

/*
 * 0  = filter (n X 1)
 * 1  = dt
 * 2  = GLM type
 * 3,4,5 are cell arrays, where each cell is an array per GPU: (if only using one GPU/block of data, these can be kcArray structs)
 * 3  = matrix X (m x n)
 * 4  = y (m x 1, vector on gpu)
 * 5  = extra space for computations
                if computing hessian (m x (n + 3))
                if not computing hessian (m x 3)
 * 6+ = anything more is in extra params
 *output, lhs
 * 0 = f;
 * 1 = g;
 * 2 = h;
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    if(nrhs < 6) {
        mexErrMsgTxt("Insufficient number of inputs.");
    }
    if(nlhs > 3) {
        mexErrMsgTxt("Too many outputs requested.");
    }
    
    //input indicies
    const int IDX_DT = 1;
    const int IDX_X  = 3;
    const int IDX_Y  = 4;
    const int IDX_C  = 5;
    const int IDX_B  = 0;
    const int IDX_TYPE  = 2;
    const int IDX_EXTRA = 6;
    
    if(!mxIsScalar(prhs[IDX_DT])) {
        mexErrMsgTxt("input 'dt' must be a scalar!");
    }
    if(!mxIsScalar(prhs[IDX_TYPE])) {
        mexErrMsgTxt("input 'GLM type' must be a scalar!");
    }
    if(!mxIsDouble(prhs[IDX_B])) {
        mexErrMsgTxt("input 'filter' must be a floating point array of correct double type!");
    }
    
    
    
    long int N = mxGetNumberOfElements(prhs[IDX_B]);
    int GLMtype = (int)mxGetScalar(prhs[IDX_TYPE]);
    double dt = mxGetScalar(prhs[IDX_DT]);    

    int nExtraParams = nrhs - IDX_EXTRA;
    double * extraParams;
    if(nExtraParams > 0) {
        extraParams = (double*)malloc(sizeof(double)*nExtraParams);
        for(int ii = 0; ii < nExtraParams; ii++) {
            extraParams[ii] = mxGetScalar(prhs[IDX_EXTRA + ii]);
        }
    }
    double * f;
    double * g;
    double * h;
    if(nlhs > 2) {
        plhs[2] = mxCreateNumericMatrix(N,N,mxDOUBLE_CLASS,mxREAL);
        h = (double*)mxGetPr(plhs[2]);
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateNumericMatrix(N,1,mxDOUBLE_CLASS,mxREAL);
        g = (double*)mxGetPr(plhs[1]); 
    }
    if(nlhs > 0) {
        plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
        f = (double*)mxGetPr(plhs[0]);
    }
    
    
    //gets FP type
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
    
    
    //runs NLL computation
    
    if(arrayType == KC_FLOAT_ARRAY) {
        computeNLL<float>(f, g, h,prhs[IDX_X], prhs[IDX_Y], prhs[IDX_C], prhs[IDX_B], N, dt,  GLMtype, extraParams,  nExtraParams,nlhs);
    }
    else if(arrayType == KC_DOUBLE_ARRAY) {
        computeNLL<double>(f, g, h,prhs[IDX_X], prhs[IDX_Y], prhs[IDX_C], prhs[IDX_B], N, dt,  GLMtype, extraParams,  nExtraParams,nlhs);
    }
    else {
        mexErrMsgTxt("Invalid GPU array inputs.\n");
    }
    
    if(nExtraParams > 0) {
        free(extraParams);
    }
    
}
