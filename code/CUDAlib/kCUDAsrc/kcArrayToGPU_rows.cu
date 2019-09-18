//#include <cuda.h>
#include "mex.h"
#include "matrix.h"
 
#include "helper_cuda.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs != 2 && nrhs != 3) {
        mexPrintf("Incorrect RHS args: expected 2 or 3and received %d (kcArrayToGPU) ", nrhs);
        
        mexErrMsgTxt("CUDA errors");
    }
    if(nlhs != 1) {
        mexPrintf("Incorrect LHS args: expected 1 and received %d (kcArrayToGPU) ", nlhs);
        
        mexErrMsgTxt("CUDA errors");
    }
    
    if(nrhs > 2) {
        int newDevice = (int)mxGetScalar(prhs[2]);
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
        mexErrMsgTxt("Data type error: input must be a single, double or Int32 matrix (kcArrayToGPU_rows) ");
    }
    
    
    if(!mxIsLogical(prhs[1])) {
        mexPrintf("Incorrect RHS args: arg 2 must be boolean (kcArrayToGPU_rows)");
        mexErrMsgTxt("CUDA errors");
    }
    if(mxGetNumberOfDimensions(prhs[0]) != 2) {
        mexPrintf("Incorrect RHS args: arg 1 must 2 dimensional matrix (kcArrayToGPU_rows)");
        mexErrMsgTxt("CUDA errors");
    }
    
    mwSize * size_0 = mxGetDimensions(prhs[0]);
    size_t size_1 = mxGetNumberOfElements(prhs[1]);
    if(size_0[0] != size_1) {
        mexPrintf("Incorrect RHS args: arg 1 must have same number of rows as arg 2 (kcArrayToGPU_rows)");
        mexErrMsgTxt("CUDA errors");
    }
    
    mxLogical * rdata = (int *)mxGetPr(prhs[1]);
    int Nrows = 0;
    for(int ii = 0; ii < size_1; ii++) {
        Nrows += rdata[ii];
    }
    
    mwSize * size_2 = mxMalloc(2 * sizeof(mwSize)); 
    size_2[0] = Nrows;
    
    
    

    plhs[0] = kcSetupEmptyArray(2,size_2,dataType);


    // get number of elements 
    size_t numElements  =mxGetNumberOfElements(prhs[0]);
    // memory size
    mwSize memSize =  elementSize  * 2*Nrows;
    
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
