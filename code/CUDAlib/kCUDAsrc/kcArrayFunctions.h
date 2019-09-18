#include <cuda.h>
#include "mex.h"
#include "matrix.h"
 

#include <kcDefs.h>

mxArray* kcSetupEmptyArray(unsigned int numDims, const mwSize * dims, int arrayType);

//gets a floating point pointer to an array's constant (Array should be double/single type!)
void * kcGetArrayData(const mxArray * arrayInfo);

void * kcGetArrayData(const mxArray * arrayInfo, unsigned int minSize);


//gets total number of elements in array
unsigned long long kcGetArrayNumEl(const mxArray * arrayInfo);

//gets size of an array along a specific dimension
long long int kcGetArraySize(const mxArray * arrayInfo, int dim);

//get device number of array
unsigned int kcGetArrayDev(const mxArray * arrayInfo);

//num dims of array (should just be 1 or 2)
unsigned int kcGetArrayNumDims(const mxArray * arrayInfo);

//changes to GPU for the given matlab gpu struct
int kcSwitchToDevice(const mxArray * arrayInfo);

//changes to specific GPU number
void kcSwitchToDevice(const int devNum);

int kcGetArrayType(const mxArray * arrayInfo);

template <class T> int getKCtype();
