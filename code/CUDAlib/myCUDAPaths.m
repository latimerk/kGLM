function [ projectHome, CUDAdirectory, CUDAhelperDirectory, MATLABdirectory ] = myCUDAPaths(  )
%MYPATHS this function contains path information to the CUDA and MATLAB folders.
%   This is used for compiling CUDA files into mex files.


%% 1. Set absolute path to the base directory for this project
projectHome = which('myCUDAPaths.m');
projectHome = projectHome(1:end-22);

% check if directory exists
if(~isfolder(projectHome))
    warning(['ERROR: projectHome directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],projectHome);
end

% IF this fails for some reason, specify absolute path to directory containing this function, e.g.:
% projectHome =  '/home/USERNAME/gitCode/GLM/';

%% 2. Set absolute path for directory where CUDA installation lives:
CUDAdirectory   = '/usr/lib/cuda/';
CUDAhelperDirectory = [CUDAdirectory '/samples/common/inc/']; %samples that come with the CUDA sdk - directory to the helper_cuda.h for error checking
               


% check if directory exists
if(~isfolder(CUDAdirectory))
    warning(['ERROR: CUDAdirectory directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDAdirectory);
end
if(~isfolder(CUDAhelperDirectory))
    warning(['ERROR: CUDASamplesdirectory directory does not exist: %s\n----\n', ...
             'Please fix by editing myPaths.m\n '],CUDAhelperDirectory);
end


%% 3. Directory of the MATLAB installation. 
MATLABdirectory = matlabroot;  % this *shouldn't* need adjusting
MATLABdirectory = [MATLABdirectory, '/'];
