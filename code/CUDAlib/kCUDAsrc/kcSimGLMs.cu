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

#include <curand.h>

#include <float.h>


__global__ void setupPSTHhelper(double * psthHelper, const int S, const int G) {
    long int ss = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int gg = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    int N = S*G;
    if(ss < N) {
        if(gg < G) {
            if(ss >= gg*S && ss < (gg+1)*S) {
                psthHelper[ss + N*gg] = 1.0;
            }
            else {
                psthHelper[ss + N*gg] = 0.0;
            }
        }
    }
}

__global__ void copySpikes(double * spks, const double * Y, const int S, const int G, const int T) {
    long int ss = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int gg = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    long int tt = (long int)blockIdx.z*(long int)blockDim.z+(long int)threadIdx.z; 
    if(tt < T) {
        if(ss < S) {
            if(gg < G) {
                int idx = gg*S + ss + tt*(G*S);
                spks[idx] = Y[gg + tt*G];
            }
        }
    }
}

__global__ void psKernelEx(double * spks, const double * rs, const double * spkHist, const double ldt, const double * X, const int S, const int G) {
    long int ss = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int gg = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(ss < S) {
        if(gg < G) {
            int idx = gg*S + ss;
            double ps = 1.0-exp(-exp(min(90.0,X[gg] + spkHist[idx] + ldt)));
            spks[idx] = (rs[idx] < ps) ? 1.0 : 0.0;
        }
    }
}
__global__ void psKernelLin(double * spks, const double * rs, const double * spkHist, const double dt, const double * X, const int S, const int G) {
    long int ss = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int gg = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(ss < S) {
        if(gg < G) {
            int idx = gg*S + ss;
            double xx = X[gg] + spkHist[idx];
            double rr = (xx < 35)?log1p(exp(xx)):xx;
            
            double ps = 1.0-exp(-rr*dt);
            spks[idx] = (rs[idx] < ps) ? 1.0 : 0.0;
        }
    }
}
__global__ void psKernelPow(double * spks, const double * rs, const double * spkHist, const double dt, const double alpha,const double * X, const int S, const int G) {
    long int ss = (long int)blockIdx.x*(long int)blockDim.x+(long int)threadIdx.x; 
    long int gg = (long int)blockIdx.y*(long int)blockDim.y+(long int)threadIdx.y; 
    if(ss < S) {
        if(gg < G) {
            int idx = gg*S + ss;
            double xx = X[gg] + spkHist[idx];
            double rr = (xx < 35)?log1p(exp(xx)):xx;
            rr = pow(rr,alpha);
            
            double ps = 1.0-exp(-rr*dt);
            spks[idx] = (rs[idx] < ps) ? 1.0 : 0.0;
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    if(nrhs < 6) {
        mexErrMsgTxt("Insufficient number of inputs.");
    }

    
    //input indicies
    const int IDX_H  = 0;
    const int IDX_DT = 1;
    
    const int IDX_A  = 2;
    
    const int IDX_X  = 3;
    const int IDX_Y0 = 4; 
    
    const int IDX_S  = 5;
    
    const int IDX_DEVNUM  = 6;
    
    
    
    if(!mxIsScalar(prhs[IDX_DT])) {
        mexErrMsgTxt("input 'dt' must be a scalar!");
    }
    if(!mxIsScalar(prhs[IDX_A])) {
        mexErrMsgTxt("input 'GLM type' must be a scalar!");
    }
    
    if(nrhs >= 7 && !mxIsScalar(prhs[IDX_DEVNUM])) {
        mexErrMsgTxt("input 'dev num' must be a scalar!");
    }
    
    int devNum = 0;
    if(nrhs >= IDX_DEVNUM+1) {
        devNum = (int)mxGetScalar(prhs[IDX_DEVNUM]);
    }

    
    checkCudaErrors(cudaSetDevice(devNum));
    
    long int T;
    long int G;
    
    long int T_0;
    long int G_0;
    
    
    int numDims_X = mxGetNumberOfDimensions(prhs[IDX_X]);
    const mwSize * dims_X = mxGetDimensions(prhs[IDX_X]);
    if(numDims_X == 1) {
        T = mxGetNumberOfElements(prhs[IDX_X]);
        G = 1;
    }
    else {
        T = dims_X[1];
        G = dims_X[0];
    }
    
    int numDims_Y = mxGetNumberOfDimensions(prhs[IDX_Y0]);
    const mwSize * dims_Y= mxGetDimensions(prhs[IDX_Y0]);
    if(numDims_Y == 1) {
        T_0 = mxGetNumberOfElements(prhs[IDX_Y0]);
        G_0 = 1;
    }
    else {
        T_0 = dims_Y[1];
        G_0 = dims_Y[0];
    }
    
    if(G != G_0 && T_0 > 0) {
        mexErrMsgTxt("input sizes do not match");
    }
    
    double * X_cpu = (double*)mxGetPr(prhs[IDX_X]);
    double * Y0_cpu = (double*)mxGetPr(prhs[IDX_Y0]);
    
    double * X;
    double * Y0;
    
    checkCudaErrors(cudaMalloc((void**)&(X),sizeof(double)*T*G));
    checkCudaErrors(cudaMemcpyAsync(X,X_cpu,sizeof(double)*T*G,cudaMemcpyHostToDevice));
    
    if(T_0 > 0) {   
        checkCudaErrors(cudaMalloc((void**)&(Y0),sizeof(double)*T_0*G));
        checkCudaErrors(cudaMemcpyAsync(Y0,Y0_cpu,sizeof(double)*T_0*G,cudaMemcpyHostToDevice));
    }
    
    int S = (int)mxGetScalar(prhs[IDX_S]);
    int N = S*G;
    
    
    int H      = mxGetNumberOfElements(prhs[IDX_H]);
    
    double * h_spk = (double*)mxGetPr(prhs[IDX_H]);
    double * h_spk2 = (double*)malloc(H*sizeof(double));
    for(int ii = 0; ii < H; ii++) {
        h_spk2[ii]   = h_spk[H-(ii+1)];
        //h_spk2[H+ii] = h_spk[H-(ii+1)];
    }
    
    double * h_gpu;
    
    
    
    checkCudaErrors(cudaMalloc((void**)&(h_gpu),sizeof(double)*H));
    checkCudaErrors(cudaMemcpyAsync(h_gpu,h_spk2,sizeof(double)*H,cudaMemcpyHostToDevice));
    
    double * spks;
    
    checkCudaErrors(cudaMalloc((void**)&(spks),sizeof(double)*N*(T+H)));
    cudaMemset(spks, 0, N*(T+H)*sizeof(double));
    
    
    double * rs;
    checkCudaErrors(cudaMalloc((void**)&(rs),sizeof(double)*N*(T)));
    
    
    double alpha = mxGetScalar(prhs[IDX_A]);
    double dt    = mxGetScalar(prhs[IDX_DT]);  
    double ldt = log(dt);  
    //=================================================

    double aa = 1;
    double bb = 0;
    
    double * spkHist;
    double * spks_c;
    double * rs_c;
    double * X_c;
    checkCudaErrors(cudaMalloc((void**)&(spkHist),sizeof(double)*T));
    
    
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&(handle));
    if(stat != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS initialization failed\n");
    }
    
    curandStatus_t stat_rand;
    curandGenerator_t gen;
    stat_rand = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    if(stat_rand != CURAND_STATUS_SUCCESS) {
        mexErrMsgTxt("CURAND initialization failed\n");
    }
    
    stat_rand = curandGenerateUniformDouble(gen, rs, T*N);
    if(stat_rand != CURAND_STATUS_SUCCESS) {
        mexErrMsgTxt("CURAND generation failed\n");
    }
    
    
    
    dim3 block_size_0;
    block_size_0.x = 128;
    block_size_0.y = 512/block_size_0.x;
    block_size_0.z = 2;
    
    dim3 grid_size_0;
    grid_size_0.x = S/block_size_0.x   + ((S%block_size_0.x == 0)? 0:1);
    grid_size_0.y = G/block_size_0.y   + ((G%block_size_0.y == 0)? 0:1);
    grid_size_0.z = T_0/block_size_0.z + ((T_0%block_size_0.z == 0)? 0:1);
    
    spks_c = spks + H*N;        
    copySpikes<<<grid_size_0,block_size_0>>>(spks_c,Y0,S,G,T_0);
    
    
    dim3 block_size;
    block_size.x = 256;
    block_size.y = 1024/block_size.x;
    
    dim3 grid_size;
    grid_size.x = S/block_size.x + ((S%block_size.x == 0)? 0:1);
    grid_size.y = G/block_size.y + ((G%block_size.y == 0)? 0:1);

    
    for(int tt = T_0; tt < T; tt++) {
            
        spks_c = spks+tt*N;

        stat =  cublasDgemv(handle,  CUBLAS_OP_N,
                       N, H,
                       &aa,
                       spks_c, N,
                       h_gpu, 1,
                       &bb,
                       spkHist, 1);

        if(stat != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("spike hist matrix times filter failed");
        }

        rs_c   = rs   + tt*N;
        spks_c = spks + (tt+H)*N;
        X_c    = X    + tt*G;

        if(alpha <= 0) {
            psKernelEx<<<grid_size,block_size>>>(spks_c,rs_c,spkHist,ldt,X_c,S,G);
        }
        else if(alpha == 1) {
            psKernelLin<<<grid_size,block_size>>>(spks_c,rs_c,spkHist,dt,X_c,S,G);
        }
        else {
            psKernelPow<<<grid_size,block_size>>>(spks_c,rs_c,spkHist,dt,alpha,X_c,S,G);
        }

    }
    
    if(nlhs > 0) {
        
        spks_c = spks + H*N;
        
        
        double * psth_helper;
        checkCudaErrors(cudaMalloc((void**)&(psth_helper),sizeof(double)*N*G));
        
        
        dim3 block_size2;
        block_size2 = 1024;
        block_size2.y = 1;

        dim3 grid_size2;
        grid_size2.x = N/block_size2.x + ((N%block_size2.x == 0)? 0:1);
        grid_size2.y = G/block_size2.y + ((G%block_size2.y == 0)? 0:1);
        
        setupPSTHhelper<<<grid_size2,block_size2>>>(psth_helper,S,G);
        
        double * psth;
        checkCudaErrors(cudaMalloc((void**)&(psth),sizeof(double)*T*G));
        
        aa = 1.0/S;
        stat =  cublasGEMM(handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       G, T, N,
                       &aa,
                       psth_helper, N,
                       spks_c, N,
                       &bb,
                       psth, G);
                if(stat != CUBLAS_STATUS_SUCCESS) {
                    mexErrMsgTxt("summary matrix multiplication failed");
                }
        
        
        
        
        plhs[0] = mxCreateNumericMatrix(G,T,mxDOUBLE_CLASS,mxREAL);
        double * a = (double*)mxGetPr(plhs[0]);
    	checkCudaErrors(cudaMemcpyAsync(a,psth,sizeof(double)*G*T,cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        checkCudaErrors(cudaFree(psth_helper));
        checkCudaErrors(cudaFree(psth));
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateNumericMatrix(N,T,mxDOUBLE_CLASS,mxREAL);
        double * f = (double*)mxGetPr(plhs[1]);
        spks_c = spks + H*N;
    	checkCudaErrors(cudaMemcpyAsync(f,spks_c,sizeof(double)*N*T,cudaMemcpyDeviceToHost));
    }
    
    //=================================================
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(h_gpu));
    checkCudaErrors(cudaFree(spks));
    checkCudaErrors(cudaFree(spkHist));
    checkCudaErrors(cudaFree(rs));
    
    if(T_0 > 0) {
        checkCudaErrors(cudaFree(Y0));
    }
    checkCudaErrors(cudaFree(X));
    
    stat = cublasDestroy(handle);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS destroy failed");
    }
    
    stat_rand = curandDestroyGenerator(gen);
    if(stat_rand != CURAND_STATUS_SUCCESS) {
        mexErrMsgTxt("CURAND destroy failed\n");
    }
    
    cudaDeviceSynchronize();
    free(h_spk2);
}
