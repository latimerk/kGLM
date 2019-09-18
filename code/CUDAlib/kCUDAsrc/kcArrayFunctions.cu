
#include "kcArrayFunctions.h"


template <> int getKCtype<float>() {
    return KC_FLOAT_ARRAY;
}
template <> int getKCtype<double>() {
    return KC_DOUBLE_ARRAY;
}
template <>int getKCtype<int>() {
    return KC_INT_ARRAY;
}

mxArray* kcSetupEmptyArray(unsigned int numDims, const mwSize * dims, int arrayType) {

    const char * field_names[] =  {KC_ARRAY_NUMEL,KC_ARRAY_NDIM,KC_ARRAY_SIZE,KC_ARRAY_PTR,KC_ARRAY_TYPE,KC_ARRAY_DEV};

    mwSize structDims[1] = {1};
    mxArray*emptyArray = mxCreateStructArray(1,structDims, 6, field_names);
    
    int numelField = mxGetFieldNumber(emptyArray,KC_ARRAY_NUMEL);
    int dimField  = mxGetFieldNumber(emptyArray,KC_ARRAY_NDIM);
    int sizeField = mxGetFieldNumber(emptyArray,KC_ARRAY_SIZE);
    int ptrField  = mxGetFieldNumber(emptyArray,KC_ARRAY_PTR);
    int typeField = mxGetFieldNumber(emptyArray,KC_ARRAY_TYPE);
    int devField = mxGetFieldNumber(emptyArray,KC_ARRAY_DEV);
    
    int currentDevice;
    cudaError_t ce;
    ce = cudaGetDevice(&currentDevice);
    if(ce != cudaSuccess) {
        mexPrintf("Error selecting device ");
        mexPrintf(cudaGetErrorString(ce));
        mexPrintf(" (%d)\n", (int)ce);
        mexErrMsgTxt("CUDA Errors");
    }
    
    mxArray* dimsArray = mxCreateNumericMatrix(1,numDims,mxINT64_CLASS,mxREAL);
    long long int * dimsPtr = (long long int*) mxGetPr(dimsArray);
    memcpy(dimsPtr,dims,sizeof(mwSize)*numDims);


    mxArray* numDimsArray = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL);
    unsigned int * numDimsPtr = (unsigned int*)mxGetPr(numDimsArray);
    numDimsPtr[0] = numDims;
    
    
    mxArray * typeVal = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL);
    unsigned int * typePtr = (unsigned int*) mxGetPr(typeVal);
    *typePtr = arrayType;


    unsigned long long numElements = 1;
    for(int i = 0; i <numDims;i++) {
        numElements = numElements*dims[i];
    }
    mxArray * numelVal = mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
    unsigned long long * numelPtr = (unsigned long long*) mxGetPr(numelVal);
    *numelPtr = (mwSize)numElements;
    
    
    mxArray * devVal = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL);
    unsigned int * devPtr = (unsigned int*) mxGetPr(devVal);
    *devPtr = currentDevice;

    mxArray* ptrVal=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
    unsigned long long int * out = (unsigned long long int *)mxGetPr(ptrVal);
    *out = (unsigned long long int)0;


    mxSetFieldByNumber(emptyArray,0,dimField,  numDimsArray);
    mxSetFieldByNumber(emptyArray,0,sizeField, dimsArray);
    mxSetFieldByNumber(emptyArray,0,ptrField,  ptrVal);
    mxSetFieldByNumber(emptyArray,0,typeField, typeVal);
    mxSetFieldByNumber(emptyArray,0,numelField, numelVal);
    mxSetFieldByNumber(emptyArray,0,devField, devVal);

    return emptyArray;
}

void * kcGetArrayData(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        void * ptr = (void *)(unsigned long long int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));

        if(ptr == KC_NULL_ARRAY) {
            mexErrMsgTxt("Array value NULL.\n");
            return 0;
        }   
        else {
            return ptr;
        }
    }  
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

void * kcGetArrayData(const mxArray * arrayInfo, unsigned int minSize) {

    void * ptr = kcGetArrayData(arrayInfo);

    unsigned int size = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_NUMEL));
    if(size >= minSize) {
        return ptr;
    }
    else {
         mexErrMsgTxt("GPU array too small.\n");
        return 0;
    }
}



unsigned long long kcGetArrayNumEl(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayType = (unsigned long long)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        if(arrayType != 0) {
            int * ptr = (int *)(unsigned long long int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));
            unsigned long long size = ((unsigned long long)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_NUMEL)));
            
            if(ptr == KC_NULL_ARRAY) {
                mexErrMsgTxt("Array value NULL.\n");
            }
            else {
                return size;
            }
        }
        else {
            mexErrMsgTxt("Invalid GPU array type.\n");
        }
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

long long int kcGetArraySize(const mxArray * arrayInfo, int dim) {
    if(mxIsStruct(arrayInfo)) {
        int arrayType = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        if(arrayType != 0) {
            int * ptr = (int *)(unsigned long long int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_PTR));
            long long int * size = ((long long int*)mxGetPr(mxGetField(arrayInfo,0,KC_ARRAY_SIZE)));
            
            if(ptr == KC_NULL_ARRAY) {
                mexErrMsgTxt("Array value NULL.\n");
            }
            else {
                return size[dim];
            }
        }
        else {
            mexErrMsgTxt("Invalid GPU array type.\n");
        }
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

int kcGetArrayType(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayDev = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_TYPE));
        return arrayDev;
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

unsigned int kcGetArrayDev(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayDev = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_DEV));
        return arrayDev;
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

unsigned int kcGetArrayNumDims(const mxArray * arrayInfo) {
    if(mxIsStruct(arrayInfo)) {
        int arrayNumDims = (unsigned int)mxGetScalar(mxGetField(arrayInfo,0,KC_ARRAY_NDIM));
        return arrayNumDims;
    }
    else {
        mexErrMsgTxt("Invalid GPU array struct.\n");
    }
    return 0;
}

int kcSwitchToDevice(const mxArray * arrayInfo) {
    int currentDevice;
    cudaGetDevice(&currentDevice);
    int devNum = kcGetArrayDev(arrayInfo);
    if(currentDevice != devNum) {
        cudaError_t ce;
        
        ce = cudaSetDevice(devNum);
        if(ce != cudaSuccess) {
            mexPrintf("Error selecting device ");
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }
        //mexPrintf("Changed to GPU device: %d (from %d)\n",devNum,currentDevice);
    }
    return devNum;
}

void kcSwitchToDevice(const int devNum) {
    int currentDevice;
    cudaGetDevice(&currentDevice);
    if(currentDevice != devNum) {
        cudaError_t ce;
        
        ce = cudaSetDevice(devNum);
        if(ce != cudaSuccess) {
            mexPrintf("Error selecting device ");
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");
        }
        //mexPrintf("Changed to GPU device: %d (from %d)\n",devNum,currentDevice);
    }
}
