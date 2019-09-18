function [GLM,nllVal_final,nllVal_init,GLMp,h_init] = fitGLMGPU_Multi(CBGLM,Xfull_gpu,space_gpu,Y_gpu,initPoint,noOptimization)
%%
% INPUTS:
%    CBGLM = structure with items needed for my complete GLM structure (note - need not be an actual CBGLM struct) 
%       contains:
%          spkHistBasisVectors  = matrix, columns are the spike history basis vectors
%          stimBasisVectors     = matrix, columns are the temporal basis vectors for the stimulus
%          dt                   = bin length
%
%
%    The following can be kcArray structures or cell arrays (assigned by kcArrayToGPU)
%    If they are kcArrays, then each is the following:
%      Xfull_gpu = (M*N)     matrix on GPU - complete design matrix for GLM
%      space_gpu = (M*(N+3)) matrix on GPU - extra calculation space, does not need to be initialized
%      Y_gpu     = (M*1)     vector on GPU - spike counts in each bin
%    If they are cell arrays, then   Xfull_gpu{ii}, space_gpu{ii}, and Y_gpu{ii} must be kcArray structures 
%      located on the same GPU and following the same rules as above. However, the dimension M can differ across
%      gpus (i.e., Xfull_gpu{ii} and Xfull_gpu{ii+1} can have different numbers of rows, but must have same column size) 
%
%    initPoint  = (N*1) initial vector for optimization - on host
%% setup nll function
if(iscell(Xfull_gpu))
    nGPUs = length(Xfull_gpu);
    if(nGPUs == 1)
        nlFunc = @(xx)kcGlmNLL_Multi(xx,CBGLM.dt,1,Xfull_gpu{1},Y_gpu{1},space_gpu{1});
    elseif(nGPUs == 2)
        nlFunc = @(xx)kcGlmNLL_Multi(xx,CBGLM.dt,2,Xfull_gpu{1},Y_gpu{1},space_gpu{1},Xfull_gpu{2},Y_gpu{2},space_gpu{2});
    else
        error('Invalid number of GPUs');
    end
else
    %nGPUs = 1;
    nlFunc = @(xx)kcGlmNLLBer_Multi(xx,CBGLM.dt,1,Xfull_gpu,Y_gpu,space_gpu);
end

if(nargin < 6) 
    noOptimization = false;
end

%% run optimizer
if(~noOptimization)
    %setup optimizer options
    opts1 = optimset('MaxIter',5,'GradObj','on','Hessian','off','Display','iter','PrecondBandWidth',inf,'LargeScale','off');
    opts2 = optimset('MaxIter',200,'GradObj','on','Hessian','on','Display','iter','TolX',1e-10,'TolFun',1e-10);%,'PrecondBandWidth',inf);

    %fminunc
    doBothOpts = false; %do a run without hessian first
    if(doBothOpts)
        GLMp = fminunc(nlFunc,initPoint,opts1);
    else
        GLMp = initPoint;
    end

    GLMp = fminunc(nlFunc,GLMp,opts2);
else
    display('Skipping optimization and returning NLL.');
    GLMp = initPoint;
end

%% sets up complete GLM structure
GLM.dt = CBGLM.dt;
GLM.spkHistBasisVectors = CBGLM.spkHistBasisVectors;
GLM.stimBasisVectors = CBGLM.stimBasisVectors;
GLM.stimBasisParams = CBGLM.stimBasisParams;
GLM.spkHistBasisParams = CBGLM.spkHistBasisParams;

h_spk_len  = size(CBGLM.spkHistBasisVectors,2);
k_stim_len = length(GLMp) - 1 - h_spk_len;

GLM.h_spk  = GLMp((end-h_spk_len+1):end);
GLM.k_stim = GLMp(1:k_stim_len);
GLM.b      = GLMp(k_stim_len+1);

tic;
[nllVal_init,~,h_init] = nlFunc(initPoint);
ftime = toc;
fprintf('Full function evaluation time: %2.3f\n',ftime);

nllVal_final = nlFunc(GLMp);
display(['Final NLL value: ' num2str(nllVal_final)]);
