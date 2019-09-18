function [GLM,nllVal_final2,converged] = fitGLMGPU_bilin3_Multi(CBGLM,Stim,spkHists,Y, centerPoint, nRank, nIters,nGPUs,llTol)
%    CBGLM = structure with items needed for my complete GLM structure (note - need not be an actual CBGLM struct) 
%       contains:
%          spkHistBasisVectors  = matrix, columns are the spike history basis vectors
%          stimBasisVectors     = matrix, columns are the temporal basis vectors for the stimulus
%          stimNumBasisVectors  = number of temporal basis vectors (convenience parameter)
%          dt                   = bin length
%
%   Assumes SpikeStm has already been convolved with temporal basis (first
%   N columns are the first pixel convolved with the N temporal basis
%   vectors)
MLOAD_ROWS_SPAT = 1024*880;
if(nRank == 3)
    MLOAD_ROWS_SPAT = 1024*100;
end

MLOAD_ROWS_TEMP = MLOAD_ROWS_SPAT;

if(nargin < 6)
    nRank = 1;
end
if(nargin < 7)
    nIters = 5;
end
if(nargin < 8)
    nGPUs = 1;
end
if(nargin < 9)
    llTol = 1e-7;
end

%% size of optimization
nTemporalBasis = CBGLM.stimNumBasisVectors;                   %number of temporal basis functions
nPixels        = size(Stim{1},2)/nTemporalBasis;             %number of spatial dimension
filtLength     = size(CBGLM.stimBasisVectors,1);              %length of temporal basis functions
NN = max(nPixels,nTemporalBasis)*nRank + 1 + size(spkHists{1},2); %max number of coefficients 
%TT = size(SpikeStm,1); %total time points




%% initialize filters
k_temp = 1e-4*randn(nTemporalBasis,nRank);
k_spat = 1e-6*randn(nPixels,nRank);
k_spat(centerPoint,1) = 1e-4;
k_spat(centerPoint+1,1) = 1e-5;
k_spat(centerPoint-1,1) = 1e-5;
k_spat(centerPoint-round(sqrt(nPixels)),1) = 1e-5;
k_spat(centerPoint+round(sqrt(nPixels)),1) = 1e-5;


for jj = 1:nRank
    k_norm = max(1e-15,sqrt(sum((k_spat(:,jj)).^2)));
    k_temp(:,jj) = k_temp(:,jj).*k_norm;
    k_spat(:,jj) = k_spat(:,jj)./k_norm;
end
%make the center point of spatial kernel positive
if(~isempty(centerPoint))
    for jj = 1:nRank
        k_sign = sign(k_spat(centerPoint,jj));
        if(k_sign ~= 0) 
            k_temp(:,jj) = k_temp(:,jj).*k_sign;
            k_spat(:,jj) = k_spat(:,jj).*k_sign;
        end
    end
end
    
meanFr = mean(Y{1});
b = log(meanFr) - log(CBGLM.dt);

h_spk = zeros(size(spkHists{1},2),1);


%% setup variables on GPUS



fullStim_gpu = cell(nGPUs,1);
Y_gpu        = cell(nGPUs,1);
space_gpu    = cell(nGPUs,1);

for ii = 1:nGPUs
    kcResetDevice(ii-1);
    
    TT_C = size(spkHists{ii},1);
    fprintf('Loading %4.2f seconds of data onto GPU %d.\n', TT_C*CBGLM.dt,ii-1);
    Y_gpu{ii}        = kcArrayToGPU(Y{ii},ii-1);
    space_gpu{ii}    = kcArrayToGPU(zeros(TT_C, NN+3),ii-1);
    fullStim_gpu{ii} = kcArrayToGPU(zeros(TT_C, NN),ii-1);
    spkHists{ii} = [ ones(TT_C,1) spkHists{ii}];
    
    totalBytes = 8*(numel(Y{ii}) + TT_C*(NN*2 + 3));
    fprintf('   %4.2fmb uploaded.\n', totalBytes/1e6);
    
    neededBytes = MLOAD_ROWS_SPAT*size(Stim{ii},2)*8;
    fprintf('   %4.2fmb needed on GPU for matrix multiplication.\n', neededBytes/1e6);
    
    
    kcRegisterDevPointer(Stim{ii},ii-1);
    kcRegisterDevPointer(spkHists{ii},ii-1);
end


%% run optimizer
t_conv_spat = nan(nIters,1);
t_conv_temp = nan(nIters,1);
t_iters     = nan(nIters,1);


%prev_nllVal_final1 = inf;
prev_nllVal_final2 = inf;

start_time = tic;
converged = -1;
for ii = 1:nIters
    t_iter = tic();
    %% temporal
    fprintf('Fitting temporal kernels... ');
    
    fprintf('convolving spatial kernel + temporal basis functions... ');
    t_conv = tic;
    for kk = 1:nGPUs
        fullStim_gpu{kk}.size(2) = numel(k_temp)+size(spkHists{kk},2);
    end
    
    A = zeros(nPixels*nTemporalBasis,nRank*nTemporalBasis);
    for jj = 1:nPixels
        for kk = 1:nRank
            A((1:nTemporalBasis) + (jj-1)*nTemporalBasis,(1:nTemporalBasis) + (kk-1)*nTemporalBasis) = diag(k_spat(jj,kk)*ones(nTemporalBasis,1)); 
        end
    end

    
%     Stim_C = SpikeStm*A ;
    if(nGPUs == 1)
        kcLoadBigMM_devPtr(nGPUs,Stim{1},A,fullStim_gpu{1},spkHists{1},MLOAD_ROWS_TEMP);
    elseif(nGPUs == 2)
        kcLoadBigMM_devPtr(nGPUs,Stim{1},A,fullStim_gpu{1},spkHists{1},MLOAD_ROWS_TEMP,Stim{2},A,fullStim_gpu{2},spkHists{2},MLOAD_ROWS_TEMP);
    end
    t_conv_spat(ii) = toc(t_conv);
    

    fprintf('optimizing... \n');
    [GLM_t,~,nllVal_init1] = fitGLMGPU_Multi(CBGLM,fullStim_gpu,space_gpu,Y_gpu,[reshape(k_temp,[],1);b;h_spk]);
    
    k_temp = reshape(GLM_t.k_stim,[],nRank);
    b = GLM_t.b;
    h_spk = GLM_t.h_spk;
    
    
    %% spatial
    fprintf('Fitting spatial kernels... ');
    
    t_conv = tic();
    fprintf('convolving temporal kernel... ');
    for kk = 1:nGPUs
        fullStim_gpu{kk}.size(2) = numel(k_spat)+size(spkHists{kk},2);
    end
    
    B = zeros(nPixels*nTemporalBasis,nRank*nPixels);
    for jj = 1:nPixels
        for kk = 1:nRank
            B(:,nPixels*(kk-1)+jj) = [zeros((jj-1)*nTemporalBasis,1);k_temp(:,kk);zeros((nPixels-jj)*nTemporalBasis,1)];
        end
    end
    
    %Stim_C = SpikeStm*B;
    if(nGPUs == 1)
        kcLoadBigMM_devPtr(nGPUs,Stim{1},B,fullStim_gpu{1},spkHists{1},MLOAD_ROWS_SPAT);
    elseif(nGPUs == 2)
        kcLoadBigMM_devPtr(nGPUs,Stim{1},B,fullStim_gpu{1},spkHists{1},MLOAD_ROWS_SPAT,Stim{2},B,fullStim_gpu{2},spkHists{2},MLOAD_ROWS_SPAT);
    end
    t_conv_temp(ii) = toc(t_conv);
    

    fprintf('optimizing... \n');
    [GLM_s,nllVal_final2,~] = fitGLMGPU_Multi(CBGLM,fullStim_gpu,space_gpu,Y_gpu,[reshape(k_spat,[],1);b;h_spk]);
    k_spat = reshape(GLM_s.k_stim,[],nRank);
    b      = GLM_s.b;
    h_spk  = GLM_s.h_spk;
    
    
    if(ii == 1)
        prev_nllVal_final2 = nllVal_init1;
    end
    
    %% normalize k_spat for identifiability
    for jj = 1:nRank
        k_norm = max(1e-15,sqrt(sum((k_spat(:,jj)).^2)));
        k_temp(:,jj) = k_temp(:,jj).*k_norm;
        k_spat(:,jj) = k_spat(:,jj)./k_norm;
    end
    
    %make the center point of spatial kernel positive
    if(~isempty(centerPoint))
        for jj = 1:nRank
            k_sign = sign(k_spat(centerPoint,jj));
            if(k_sign ~= 0) 
                k_temp(:,jj) = k_temp(:,jj).*k_sign;
                k_spat(:,jj) = k_spat(:,jj).*k_sign;
            end
        end
    end
    
    t_iters(ii) = toc(t_iter);

    totalNllChange = max(0, prev_nllVal_final2 - nllVal_final2);
    
    fprintf('Iter %d / %d complete. Total nll change: %2.10f.  Iteration time: %3.2f\n\tconvolutions     :  temp = %3.2f, spat = %3.2f, total = %3.2f\n', ii, nIters,totalNllChange,t_iters(ii),t_conv_temp(ii), t_conv_spat(ii),t_conv_temp(ii) + t_conv_spat(ii));
    
    if(totalNllChange < llTol)
        fprintf('Change in NLL value below tolerance. Terminating.\n');
        converged = nIters;
        
        break;
    end

    prev_nllVal_final2 = nllVal_final2;
    
end

end_time = toc(start_time);
fprintf('Mean convolution times : temporal = %2.2f, spatial = %2.2f\n',nanmean(t_conv_temp),nanmean(t_conv_spat));
fprintf('Mean iteration time: %2.2f\n',nanmean(t_iters));
fprintf('Total time = %2.2f\n',end_time);

%% setup final GLM struct
GLM = GLM_s;
GLM.k_stim_t = k_temp;
GLM.k_stim_s = k_spat;
GLM.h_spk    = h_spk;
GLM.b        = b;

GLM.filt = zeros(filtLength,nPixels);
for ii = 1:nRank
    GLM.filt = (GLM.stimBasisVectors*GLM.k_stim_t(:,ii))*GLM.k_stim_s(:,ii)' + GLM.filt;
end

%% clear GPU memory
for ii = 1:nGPUs
    kcFreeGPUArray(Y_gpu{ii});
    kcFreeGPUArray(fullStim_gpu{ii});
    kcFreeGPUArray(space_gpu{ii})
    %kcFreeGPUArray(Stim_GPU{ii})
    
    kcUnregisterDevPointer(Stim{ii},ii-1);
    kcUnregisterDevPointer(spkHists{ii},ii-1);
    
    kcResetDevice(ii-1);
end
clear Stim
