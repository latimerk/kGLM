//#include "cuda/cuda.h"
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "mex.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    if(nrhs != 1) {
        mexPrintf("Incorrect RHS args: expected 1 and received %d (kcArrayToHost) ", nrhs);
        mexErrMsgTxt("CUDA errors");
    }
    if(nlhs != 1) {
        mexPrintf("Incorrect LHS args: expected 1 and received %d (kcArrayToHost) ", nlhs);
        mexErrMsgTxt("CUDA errors");
    }
    
    
    kcSwitchToDevice(prhs[0]);

	//init data
    unsigned long long mSize = kcGetArrayNumEl(prhs[0]);
    unsigned int ndims       = kcGetArrayNumDims(prhs[0]) ;
    const mwSize* size       = (const mwSize*)mxGetPr(mxGetField(prhs[0],0,KC_ARRAY_SIZE));

    
    int arrayType = kcGetArrayType(prhs[0]);
    
    int elementSize = 4;
    if(arrayType == KC_DOUBLE_ARRAY) {
        elementSize = sizeof(double);
        plhs[0] = mxCreateNumericArray(ndims,size,mxDOUBLE_CLASS,mxREAL);
    }
    else if(arrayType == KC_FLOAT_ARRAY) {
        elementSize = sizeof(float);
        plhs[0] = mxCreateNumericArray(ndims,size,mxSINGLE_CLASS,mxREAL);
    }
    else if(arrayType == KC_INT_ARRAY) {
        elementSize = sizeof(int);
        plhs[0] = mxCreateNumericArray(ndims,size,mxINT32_CLASS,mxREAL);
    }
    else {
        mexPrintf("Invalid GPU array type\n");
        return;
    }

    unsigned long long memSize = mSize*elementSize;
    
    void *d_a = kcGetArrayData(prhs[0]);
    if(d_a == KC_NULL_ARRAY) {
        mexPrintf("Invalid GPU array\n");
        return;
    }

	
    void * ans = (void*)mxGetData(plhs[0]);
    cudaError_t copyResult = cudaMemcpy(ans,d_a,memSize,cudaMemcpyDeviceToHost);
    
    if(copyResult == cudaErrorInvalidValue) {
        mexPrintf("cudaErrorInvalidValue\n");
    }
    else if(copyResult == cudaErrorInvalidDevicePointer) {
        mexPrintf("cudaErrorInvalidDevicePointer\n");
    }
    else if(copyResult == cudaErrorInvalidMemcpyDirection) {
        mexPrintf("cudaErrorInvalidMemcpyDirection\n");
    }



}
