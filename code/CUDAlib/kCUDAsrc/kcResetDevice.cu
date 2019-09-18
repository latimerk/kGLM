
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>



#include "mex.h"

#include "kcDefs.h"
#include "kcArrayFunctions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    cudaError_t ce;
    /*ce = cudaSetDevice(0);
    if(ce != cudaSuccess) {
        mexPrintf("Error selecting device: %d\n", (int)ce);
    }
    else {*/
    if(nrhs > 0) {
        int newDevice = (int)mxGetScalar(prhs[0]);
        kcSwitchToDevice(newDevice);
    }
    
    cudaGetLastError();
        ce = cudaDeviceReset();
        if(ce != cudaSuccess) {
            mexPrintf("Error reseting device: %d\n", (int)ce);
        }
        else {
            int printOutput = 0;
            if(nrhs > 1) {
                printOutput = (int)mxGetScalar(prhs[1]);
            }
            if(printOutput > 0) {
                mexPrintf("Device reset.\n");
            }
        }
    //}
}
