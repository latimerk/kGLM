#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>



#include "mex.h"



#include <float.h>

#include "mkl.h"


void setupPSTHhelper(double * psthHelper, const int S, const int G) {
    int N = S*G;
    for(int gg = 0; gg < G gg++) {
        for(int ss = 0; ss < N; ss++) {
            if(ss >= gg*S && ss < (gg+1)*S) {
                psthHelper[ss + N*gg] = 1.0;
            }
            else {
                psthHelper[ss + N*gg] = 0.0;
            }
        }
    }
}

void copySpikes(double * spks, const double * Y, const int S, const int G, const int T) {

    for(int tt = 0; tt < T; tt++) {
        for(int gg = 0; gg < G gg++) {
            for(int ss = 0; ss < S; ss++) {
                int idx = gg*S + ss + tt*(G*S);
                spks[idx] = Y[gg + tt*G];
            }
        }
    }
}

void psKernelEx(double * spks, const double * rs, const double * spkHist, const double ldt, const double * X, const int S, const int G) {

    for(int gg = 0; gg < G gg++) {
        for(int ss = 0; ss < S; ss++) {
            int idx = gg*S + ss;
            double ps = 1.0-exp(-exp(min(90.0,X[gg] + spkHist[idx] + ldt)));
            spks[idx] = (rs[idx] < ps) ? 1.0 : 0.0;
        }
    }
}
void psKernelLin(double * spks, const double * rs, const double * spkHist, const double dt, const double * X, const int S, const int G) {

    for(int gg = 0; gg < G gg++) {
        for(int ss = 0; ss < S; ss++) {
            int idx = gg*S + ss;
            double xx = X[gg] + spkHist[idx];
            double rr = (xx < 35)?log1p(exp(xx)):xx;
            
            double ps = 1.0-exp(-rr*dt);
            spks[idx] = (rs[idx] < ps) ? 1.0 : 0.0;
        }
    }
}
void psKernelPow(double * spks, const double * rs, const double * spkHist, const double dt, const double alpha,const double * X, const int S, const int G) {
    
    for(int gg = 0; gg < G gg++) {
        for(int ss = 0; ss < S; ss++) {
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
    
    
    
    double alpha = mxGetScalar(prhs[IDX_A]);
    
    
    if(!mxIsScalar(prhs[IDX_DT])) {
        mexErrMsgTxt("input 'dt' must be a scalar!");
    }
    if(!mxIsScalar(prhs[IDX_A])) {
        mexErrMsgTxt("input 'GLM type' must be a scalar!");
    }
    

    
    
    
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
    
    double * X = (double*)mxGetPr(prhs[IDX_X]);
    double * Y0 = (double*)mxGetPr(prhs[IDX_Y0]);

    int S = (int)mxGetScalar(prhs[IDX_S]);
    int N = S*G;
    
    
    int H      = mxGetNumberOfElements(prhs[IDX_H]);
    
    double * h_spk = (double*)mxGetPr(prhs[IDX_H]);
    double * h_spk2 = (double*)malloc(H*sizeof(double));
    for(int ii = 0; ii < H; ii++) {
        h_spk2[ii]   = h_spk[H-(ii+1)];
    }
    
    
    double * spks = (double*)mkl_calloc(N*(T+H),sizeof(double),64);
    
    
    
    double dt    = mxGetScalar(prhs[IDX_DT]);  
    double ldt = log(dt);  
    //=================================================

    double aa = 1;
    double bb = 0;
    
    double * spkHist = (double*)mkl_malloc(sizeof(double)*T,64);
    double * spks_c;
    double * rs_c;
    double * X_c;
    
    
    //============================================
    
    double * rs = (double*)mkl_malloc(sizeof(double)*N*(T),64);
    VSLStreamStatePtr stream;
    vslNewStream( &stream, VSL_BRNG_MT19937, (int)time(NULL) );
        
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N*T, rs, 0.0, 1.0 );

    vslDeleteStream( &stream );
    //=============================================
    

    
    spks_c = spks + H*N;        
    copySpikes(spks_c,Y0,S,G,T_0);
    
    
    
    for(int tt = T_0; tt < T; tt++) {
            
        spks_c = spks+tt*N;

        cblas_dgemv(CblasColumnMajor,  CblasNoTrans,
                       N, H,
                       &aa,
                       spks_c, N,
                       h_spk2, 1,
                       &bb,
                       spkHist, 1);

        rs_c   = rs   + tt*N;
        spks_c = spks + (tt+H)*N;
        X_c    = X    + tt*G;

        if(alpha <= 0) {
            psKernelEx(spks_c,rs_c,spkHist,ldt,X_c,S,G);
        }
        else if(alpha == 1) {
            psKernelLin(spks_c,rs_c,spkHist,dt,X_c,S,G);
        }
        else {
            psKernelPow(spks_c,rs_c,spkHist,dt,alpha,X_c,S,G);
        }
    }
    
    if(nlhs > 0) {
        
    }
    if(nlhs > 1) {
        
        spks_c = spks + H*N;
        
        
        double * psth_helper = (double*)mkl_malloc(sizeof(double)*N*G,64);

        
        setupPSTHhelper(psth_helper,S,G);
        
        plhs[1] = mxCreateNumericMatrix(G,T,mxDOUBLE_CLASS,mxREAL);
        double * psth = (double*)mxGetPr(plhs[1]);
        
        aa = 1.0/S;
        cblas_dgemm(CblasColumnMajor,
                       CblasTrans, CblasNoTrans,
                       G, T, N,
                       &aa,
                       psth_helper, N,
                       spks_c, N,
                       &bb,
                       psth, G);
        
        
        mkl_free(psth_helper);
    }

    
    //=================================================
    
    
    
    mkl_free(h_spk2);
    mkl_free(spks);
    mkl_free(spkHist);
    mkl_free(rs);
}
