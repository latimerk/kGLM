function [fNames] = linkCUDAmex(fName)
if(nargin > 0)

            
    [projectHome, CUDAdirectory, ~,~] = myCUDAPaths();
    objDir    = [projectHome '/CUDAlib/kCUDAobj/'];
    mexDir    = [projectHome '/CUDAlib/kCUDAmex/'];
    
    if(~exist([objDir '/' fName '.o'],'file'))
        error('No object file found for linking! %s\n',[objDir '/' fName '.o']);
    end
    if(~isfolder(mexDir))
        mkdir(mexDir);
        addpath(mexDir);
    end
    
    if(~exist([objDir '/kcArrayFunctions.a'],'file'))
        fprintf('Compiling array function library...\n');
        compileCUDAmex('kcArrayFunctions',true);
        fprintf('Done.\n');
    end
    
    linkCUDAlibMex    = @(fName) mex('-cxx','-output', [mexDir '/' fName],'-v', '-lstdc++',['-L' CUDAdirectory '/lib64/'], '-lcuda', '-lcudart', '-lnppc', '-lnpps',  '-lcusparse', '-lcublas', '-lcusolver', '-lcurand','-lmwblas',[objDir '/' fName '.o'], [objDir '/kcArrayFunctions.a']);
    %'-lnppi',
            
    linkCUDAlibMex(fName);
    
    fNames = [mexDir '/' fName '.' mexext];
else
    fNames = compileCUDAmex();
end



