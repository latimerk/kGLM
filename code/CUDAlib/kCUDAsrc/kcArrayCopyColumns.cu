

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>


#include "mex.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"


/* copies the j indexed columns in A to B
 * 0 = matrix A (m x n) on GPU - source
 * 1 = matrix B (m x k) on GPU - target
 * 2 = int vector j (k x 1) on CPU
 *output, lhs
 * none
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    if(nrhs != 3 || nlhs != 0) {
        mexPrintf("lhs args = %d (should be 3), rhs args = %d (should be 0)\n",nlhs,nrhs); 
        mexErrMsgTxt("Incorrect number of input/output params.");
    }
    //mexPrintf("init (M)");
    int devNum = kcGetArrayDev(prhs[0]);
    int devNum2 = kcGetArrayDev(prhs[1]);
    checkCudaErrors(cudaSetDevice(devNum));
    
    if(devNum != devNum2) {
        mexErrMsgTxt("Arrays are not on the same device!");
    }
    
    long int M  = kcGetArraySize(prhs[0], 0);
    long int M2 = kcGetArraySize(prhs[1], 0);
    long int N = kcGetArraySize(prhs[0], 1);
    long int K = kcGetArraySize(prhs[1], 1);
    long int K2 = mxGetNumberOfElements(prhs[2]);
    
    
    
    if(K2 > K) {
        mexErrMsgTxt("Not enough space for result (columns)!");
    }
    if(M > M2) {
        mexErrMsgTxt("Not enough space for result (rows)!");
    }
    
    const int * cols = (int*)mxGetPr(prhs[2]);
    
    int arrayType = kcGetArrayType(prhs[0]);
    int arrayType2 = kcGetArrayType(prhs[1]);
    if(arrayType != arrayType2) {
        mexErrMsgTxt("Source and target are not the same type!");
    }
    
    if(arrayType == KC_DOUBLE_ARRAY) {
        double * target = (double*)kcGetArrayData(prhs[1]);
        const double * source = (double*)kcGetArrayData(prhs[0]);
        for(int ii = 0; ii < K2; ii++) {

            if(cols[ii] < 0 || cols[ii] >= N) {
                mexErrMsgTxt("Invalid column index for copy!");
            }
            double * target_c = target + ii*M2;
            const double * source_c = source + cols[ii]*M;
            checkCudaErrors(cudaMemcpyAsync(target_c, source_c, M *sizeof(double),  cudaMemcpyDeviceToDevice));
        }
    }
    else if(arrayType == KC_FLOAT_ARRAY) {
        float * target = (float*)kcGetArrayData(prhs[1]);
        const float * source = (float*)kcGetArrayData(prhs[0]);
        for(int ii = 0; ii < K2; ii++) {

            if(cols[ii] < 0 || cols[ii] >= N) {
                mexErrMsgTxt("Invalid column index for copy!");
            }
            float * target_c = target + ii*M2;
            const float * source_c = source + cols[ii]*M;
            checkCudaErrors(cudaMemcpyAsync(target_c, source_c, M *sizeof(float),  cudaMemcpyDeviceToDevice));
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
}