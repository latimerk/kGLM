function [fNames] = compileCUDAmex(fName,isLib)

if(nargin > 0)
    [projectHome, CUDAdirectory, CUDAhelperDir, MATLABdirectory] = myCUDAPaths();
    sourceDir = [projectHome '/CUDAlib/kCUDAsrc'];
    objDir    = [projectHome '/CUDAlib/kCUDAobj'];
    
    if(nargin < 2)
        isLib = false;
    end
    
    extraArgs = '-O2 -Xcompiler -fPIC';
    
    if(~isfolder(sourceDir))
        error('CUDA source directory not found! %s\n',sourceDir);
    end 
    if(~isfolder(objDir))
        mkdir(objDir);
    end
    
    if(isLib)
        compileFlag = '--lib';
        fExtension = '.a';
    else
        compileFlag = '-c';
        fExtension = '.o';
    end
    
    compileCUDAlibMex = @(fName) system(['cd ' objDir '; ' CUDAdirectory '/bin/nvcc ' compileFlag ' -shared -m64 --gpu-architecture sm_61 ' extraArgs ' -I' sourceDir ' -I' CUDAhelperDir ' -I' MATLABdirectory '/extern/include '  sourceDir '/' fName '.cu' ' -o ' fName fExtension]);
    
    
    compileCUDAlibMex(fName);
    
    fNames = [objDir '/'  fName '.o'];
else
    %display('NOTE: if receveing warning about /usr/local/cuda-7.0/samples/common/inc/exception.h')
    %display(' everything is fine. There is something in a CUDA lib that the compiler isn''t happy about, but it will run anyway');

    %% ========================================================================


    fNames = {'kcResetDevice'};
    fNames = {'kcSetDevice',fNames{:}}; %#ok<*CCAT>
    fNames = {'kcArrayToGPU',fNames{:}};
    fNames = {'kcArrayToGPUint',fNames{:}};
    fNames = {'kcArrayToHost',fNames{:}};
    fNames = {'kcFreeGPUArray',fNames{:}};


    fNames = {'kcGlmNLL_Multi',fNames{:}};
end
