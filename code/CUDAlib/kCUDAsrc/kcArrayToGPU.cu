//#include <cuda.h>
#include "mex.h"
#include "matrix.h"
 
#include "helper_cuda.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 1 && nrhs != 2) {
        mexPrintf("Incorrect RHS args: expected 1 or 2 and received %d (kcArrayToGPU) ", nrhs);
        
        mexErrMsgTxt("CUDA errors");
    }
    if(nlhs != 1) {
        mexPrintf("Incorrect LHS args: expected 1 and received %d (kcArrayToGPU) ", nlhs);
        
        mexErrMsgTxt("CUDA errors");
    }
    
    if(nrhs > 1) {
        int newDevice = (int)mxGetScalar(prhs[1]);
        kcSwitchToDevice(newDevice);
    }
    
    
    cudaError_t ce;
    void * idata = (void *)mxGetPr(prhs[0]);
    int elementSize;
    int dataType;
    // get idata(uint32) 
    if(mxIsDouble(prhs[0])) {
        dataType = KC_DOUBLE_ARRAY; 
        elementSize= sizeof(double);
    }
    else if(mxIsSingle(prhs[0])) {
        dataType = KC_FLOAT_ARRAY; 
        elementSize= sizeof(float);
    }
    else if(mxIsInt32(prhs[0])) {
        dataType = KC_INT_ARRAY;
        elementSize= sizeof(int); 
    }
    else {
        mexErrMsgTxt("Data type error: input must be a single, double or Int32 matrix (kcArrayToGPU) ");
    }

    plhs[0] = kcSetupEmptyArray(mxGetNumberOfDimensions(prhs[0]),mxGetDimensions(prhs[0]),dataType);


    // get number of elements 
    size_t numElements=mxGetNumberOfElements(prhs[0]);
    // memory size
    mwSize memSize =  elementSize  * numElements;
    
    size_t availableSpace;
    size_t totalSpace;
    
    checkCudaErrors(cudaMemGetInfo(&availableSpace,&totalSpace));
    //mexPrintf(" request space = %llu, available space = %llu (total space = %llu)\n", memSize, availableSpace, totalSpace);
    if(memSize > availableSpace) {
        mexPrintf("Error copying array (kcArrayToGPU) ");
        mexPrintf(" request space = %llu is greater than available space = %llu (total space = %llu)\n", memSize, availableSpace, totalSpace);
        mexErrMsgTxt("CUDA errors");
    }
    

    

    // allocate memory in GPU
    void *gdata;
    ce =  cudaMalloc( (void**) &gdata, memSize);
    if(ce != cudaSuccess) {
        mexPrintf("Error allocating array (kcArrayToGPU) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

    // copy to GPU
    ce =  cudaMemcpy( gdata, idata, memSize, cudaMemcpyHostToDevice) ;
    if(ce != cudaSuccess) {
        mexPrintf("Error copying array (kcArrayToGPU) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }


    // create MATLAB ref to GPU pointer
    /*mxArray* ptrVal=mxCreateNumericMatrix(1,1,KC_PTR_SIZE_MATLAB,mxREAL);
    unsigned KC_PTR_SIZE  int * out = (unsigned KC_PTR_SIZE int *)mxGetPr(ptrVal);
    *out = (unsigned KC_PTR_SIZE int)gdata;
    mxSetField(plhs[0],0,KC_ARRAY_PTR, ptrVal);*/
    unsigned long long * ptr = (unsigned long long int*)mxGetPr(mxGetField(plhs[0],0,KC_ARRAY_PTR));
    *ptr = (unsigned long long int)gdata;


    ce = cudaDeviceSynchronize();
    if(ce != cudaSuccess) {
        mexPrintf("Error finalizing new array allocation (kcArrayToGPU) ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA errors");
    }

}
