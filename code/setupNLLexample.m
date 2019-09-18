%% 
% Inputs
%   X  = Design matrix of size N by M
%   Y  = observation vector of length N
%   dt = bin size in seconds (or whatever...)
%
%   chose from the GLMtype variables to select the rate function
%%


N = size(X,1);
M = size(X,2);

nDevices = 2; %number of GPUs. There are 2 on the Titan Z

X_gpu = cell(nDevices,1);
Y_gpu = cell(nDevices,1);
C_gpu = cell(nDevices,1); %extra space for computations

tts = cell(nDevices,1);

NperGPU = round(N/nDevices);

for ii = 1:nDevices
    kcResetDevice(ii-1);
    endT    = ii*NperGPU;
    if(ii == nDevices)
        endT = N;
    end
    tts{ii} = ((ii-1)*NperGPU+1):endT;
    
    X_gpu{ii} = kcArrayToGPU(X(tts{ii},:),ii-1);
    Y_gpu{ii} = kcArrayToGPU(Y(tts{ii},:),ii-1);
    C_gpu{ii} = kcArrayToGPU(zeros(length(tts{ii}),M+3),ii-1);
    %note that if you are never computing Hessians of the GLM, use the definition of C_gpu below to save GPU memory
    %C_gpu{ii} = kcArrayToGPU(zeros(length(tts{ii}),3),ii-1);
end

GLMargs = 1;
% GLMtype = 0; %exponential nonlinearity, Poisson distribution for bins
% GLMtype = 1; %exponential nonlinearity, Bernoulli distribution for bins
% GLMtype = 2; %soft-rec nonlinearity, Bernoulli distribution for bins
% GLMtype = 4; %exponential nonlinearity, Poisson distribution for bins
% GLMtype = 6;
% GLMargs = 2;
% GLMargs = 3; %soft-rec to a power nonlinearity. Bernoulli distribution for bins




nllFunc = @(weights) kcGlmNLL_Multi(weights,dt,GLMtype,X_gpu,Y_gpu,C_gpu,GLMargs);
npFunc = @(weights) nllGLM_l2_GPU(weights,nllFunc,lambda);

%% when finished, clear the GPU variables with the following code
%
% for ii = 1:nDevices
%    kcFreeGPUArray(X_gpu{ii})
%    kcFreeGPUArray(Y_gpu{ii})
%    kcFreeGPUArray(C_gpu{ii})
% end


