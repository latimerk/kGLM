#ifndef   _KCUDA_CONSTS
#define   _KCUDA_CONSTS


#include <cuda_runtime.h>
#include "cublas_v2.h"

// #define KC_IS_MATLAB_FPARRAY mxIsDouble
// #define KC_FP_TYPE_MATLAB mxDOUBLE_CLASS
// #define KC_FP_TYPE_MATLAB mxSINGLE_CLASS
// #define KC_IS_MATLAB_FPARRAY mxIsSingle

//Array types
#define KC_DOUBLE_ARRAY 1
#define KC_INT_ARRAY    2
#define KC_FLOAT_ARRAY  3

//Struct field names
#define KC_ARRAY_NUMEL "numel"
#define KC_ARRAY_NDIM  "ndim"
#define KC_ARRAY_SIZE  "size"
#define KC_ARRAY_PTR   "ptr"
#define KC_ARRAY_TYPE  "type"
#define KC_ARRAY_DEV  "device"

#define KC_HANDLE_SPARSE_PTR   "sparseHandlePtr"
#define KC_HANDLE_RAND_PTR   "randHandlePtr"

#define KC_NULL_ARRAY   0


#define KC_GPU_DEVICE 0


inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc) {
    return cublasDgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline cublasStatus_t cublasGEMM(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *B, int ldb,
                           const float          *beta,
                           float          *C, int ldc) {
    return cublasSgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double          *y, int incy) {
    return cublasDgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}
inline cublasStatus_t cublasGEMV(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float          *alpha,
                           const float          *A, int lda,
                           const float          *x, int incx,
                           const float          *beta,
                           float          *y, int incy) {
    return cublasSgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}



inline __device__ float safeExp(float x) {
    return exp(min(88.7f,x));
}
inline __device__ double safeExp(double x) {
    return exp(min(709.78,x));
}
/*
inline __device__ float log(float x) {
    return logf(x);
}
inline __device__ float sqrt(float x) {
    return sqrtf(x);
}
inline __device__ float max(float x, float y) {
    return maxf(x,y);
}
inline __device__ float min(float x, float y) {
    return minf(x,y);
}
inline __device__ float log1p(float x) {
    return log1pf(x);
}
inline __device__ float expm1(float x) {
    return expm1f(x);
}
inline __device__ float pow(float x, float y) {
    return powf(x,y);
}*/


#endif



