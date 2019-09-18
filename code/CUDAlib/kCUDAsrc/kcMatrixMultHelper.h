//=============================================================================================================================================
//=============================================================================================================================================
//==========================================                                                     ==============================================
//==========================================                       matrix mullt                  ==============================================
//==========================================                                                     ==============================================
//=============================================================================================================================================
//=============================================================================================================================================

template <class T>
__global__ void copyHHelper(T * H_ans,const int X, const int Y,const int maxC, const int N) {
    int row = blockIdx.x*blockDim.x+threadIdx.x; 
    if(row < maxC && Y+row < N) {
        for(int cc = 0; cc < maxC && cc+X < N;cc++) {
            int origin = (Y+row)*N+cc+X;
            int dest   = (cc+X)*N + (Y+row);
            
            H_ans[dest] = H_ans[origin];
        }
    }
}
template 
__global__ void copyHHelper<float>(float * H_ans,const int X, const int Y,const int maxC, const int N);
template 
__global__ void copyHHelper<double>(double * H_ans,const int X, const int Y,const int maxC, const int N);

template <class T>
void matrixMultiplyHelper(T * X, T * C, T * H_ans,int N,int N2 , int M,T * alpha,T * beta,cudaStream_t stream,cublasHandle_t handle,cublasOperation_t OP1,cublasOperation_t OP2, int ldH) {
    int GEMM_MAX_COLS  = 192; 
    cublasStatus_t stat;
    cublasSetStream(handle, stream);
    //int ctr = 0;
    int ldx = M;
    if(OP1 != CUBLAS_OP_T) {
        ldx = N;
    }
    int ldc = M;
    if(OP2 != CUBLAS_OP_N) {
        ldc = N2;
    }
    
    
    if(N <= GEMM_MAX_COLS) {
        stat =  cublasGEMM(handle,
                       OP1, OP2,
                       N, N2, M,
                       alpha,
                       X, ldx,
                       C, ldc,
                       beta,
                       H_ans, ldH);
    }
    else { //For some reason cublas has troubles managing memory with big matrices - this multiple-run version is faster in many of those cases
        int SI_Y_start;
        int SYMM = (X == C);
        for(int SI_X = 0; SI_X < N; SI_X += GEMM_MAX_COLS) {
            if(SYMM) {
                SI_Y_start = SI_X;
            }
            else {
                SI_Y_start = 0;
            }
            for(int SI_Y = SI_Y_start; SI_Y < N2; SI_Y += GEMM_MAX_COLS) {//N2 or N+1???
                //printf("Launching GEMM %d...\n",ctr);

                int CR_X  = N -SI_X;
                int CR_Y  = N2-SI_Y;

                int nCols_X = min(GEMM_MAX_COLS,CR_X);
                int nCols_Y = min(GEMM_MAX_COLS,CR_Y);

                T * X_start;//
                if(OP1 == CUBLAS_OP_T) {
                    X_start = X + SI_X*ldx;
                }
                else {
                    X_start = X + SI_X;
                }
                T * C_start;
                if(OP2 == CUBLAS_OP_N) {
                    C_start = C + SI_Y*ldc;
                }
                else {
                    C_start = C + SI_Y;
                }
                T * H_start = H_ans + SI_X + SI_Y*ldH;

                stat =  cublasGEMM(handle,
                                       OP1, OP2,
                                       nCols_X, nCols_Y, M,
                                       alpha,
                                       X_start, ldx,
                                       C_start, ldc,
                                       beta,
                                       H_start, ldH);
                if(stat != CUBLAS_STATUS_SUCCESS) {
                    mexErrMsgTxt("matrix multiplication failed");
                }
                //ctr++;

                if(SYMM && SI_Y != SI_X) {
                    copyHHelper<T><<<GEMM_MAX_COLS,1,0,stream>>>(H_ans,SI_X, SI_Y,GEMM_MAX_COLS, ldH);
                }
            }
        }
    }
}
template void matrixMultiplyHelper<float>(float * X, float * C, float * H_ans,int N,int N2 , int M,float * alpha,float * beta,cudaStream_t stream,cublasHandle_t handle,cublasOperation_t OP1,cublasOperation_t OP2, int ldH);
template void matrixMultiplyHelper<double>(double * X,double * C, double * H_ans,int N,int N2 , int M,double * alpha,double * beta,cudaStream_t stream,cublasHandle_t handle,cublasOperation_t OP1,cublasOperation_t OP2, int ldH);



template <class T>
void matrixMultiplyHelper(T * X, T * C, T * H_ans,int N,int N2 , int M,T * alpha,T * beta,cudaStream_t stream,cublasHandle_t handle,cublasOperation_t OP1,cublasOperation_t OP2) {
    matrixMultiplyHelper(X, C, H_ans,N,N2 , M,alpha,beta,stream, handle, OP1, OP2,N);
}
template void matrixMultiplyHelper<float>(float * X, float * C, float * H_ans,int N,int N2 , int M,float * alpha,float * beta,cudaStream_t stream,cublasHandle_t handle,cublasOperation_t OP1,cublasOperation_t OP2);
template void matrixMultiplyHelper<double>(double * X, double * C, double * H_ans,int N,int N2 , int M,double * alpha,double * beta,cudaStream_t stream,cublasHandle_t handle,cublasOperation_t OP1,cublasOperation_t OP2);

template <class T>
void matrixMultiplyHelper(T * X, T * C, T * H_ans,int N,int N2 , int M,T * alpha,T * beta,cudaStream_t stream,cublasHandle_t handle) {
    matrixMultiplyHelper(X, C, H_ans, N, N2 , M, alpha, beta, stream, handle,CUBLAS_OP_T,CUBLAS_OP_N);
}
template void matrixMultiplyHelper<float>(float * X, float * C, float * H_ans,int N,int N2 , int M,float * alpha,float * beta,cudaStream_t stream,cublasHandle_t handle);
template void matrixMultiplyHelper<double>(double * X, double * C, double * H_ans,int N,int N2 , int M,double * alpha,double * beta,cudaStream_t stream,cublasHandle_t handle);