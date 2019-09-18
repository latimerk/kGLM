M = 10;     %num dimensions
N = 3200e1; %num observations

GLMtype = 1;
GLMargs = 2;

phi_true = 0.4;

w_true = randn(M,1)*0.1;
w_true(rand(M,1) > phi_true) = 0;
s_true = zeros(M,1);
s_true(w_true ~= 0) = 1;
X = randn(N,M);

Xl = X*w_true;

lambda = @(xx) powFunc(xx,GLMargs);

y = min(1,poissrnd(lambda(Xl)));


N2 = 10e6;
X_pred = randn(N2,M);
y_pred = poissrnd(exp(X_pred*w_true));
%%nllFunc_pred = @(weights) nllGLM(weights,X_pred,y_pred);

M_max = M; %info for sampler - sample all dims at once
Xb_gpu = [];
XP_gpu = [];

%%
nDevices = 2;
nFolds   = nDevices;


X_gpu = cell(nFolds,1);
Y_gpu = cell(nFolds,1);
C_gpu = cell(nFolds,1); %extra space for computations

tts = cell(nFolds,1);

NperFold = round(N/nFolds);

for ii = 1:nFolds
    deviceNum = mod(ii-1,nDevices);
    if(ii <= nDevices)
        kcResetDevice(deviceNum);
    end
    endT    = ii*NperFold;
    if(ii == nFolds)
        endT = N;
    end
    tts{ii} = ((ii-1)*NperFold+1):endT;
    
    X_gpu{ii} = kcArrayToGPU(X(tts{ii},:),deviceNum);
    Y_gpu{ii} = kcArrayToGPU(y(tts{ii},:),deviceNum);
    C_gpu{ii} = kcArrayToGPU(zeros(length(tts{ii}),3 + M*2),deviceNum);
end


nllFunc_gpu = @(weights) kcGlmNLL_Multi(weights,1,GLMtype,X_gpu,Y_gpu,C_gpu,GLMargs);
nllFunc_cpu = @(weights) nllGLM(weights,X,y);


%%

X_pred_gpu = cell(nDevices,1);
Y_pred_gpu = cell(nDevices,1);
C_pred_gpu = cell(nDevices,1); %extra space for computations

tts_pred = cell(nDevices,1);

NperFold = round(size(X_pred,1)/nDevices);

for ii = 1:nDevices
    %kcResetDevice(ii-1);
    endT    = ii*NperFold;
    if(ii == nDevices)
        endT = size(X_pred,1);
    end
    tts_pred{ii} = ((ii-1)*NperFold+1):endT;
    
    X_pred_gpu{ii} = kcArrayToGPU(X_pred(tts_pred{ii},:),ii-1);
    Y_pred_gpu{ii} = kcArrayToGPU(y_pred(tts_pred{ii},:),ii-1);
    C_pred_gpu{ii} = kcArrayToGPU(zeros(length(tts_pred{ii}),M+3),ii-1);
end

nllFunc_pred = @(weights) kcGlmNLL_Multi(weights,1,GLMtype,X_pred_gpu,Y_pred_gpu,C_pred_gpu,GLMargs);

%w_mle = w_true;

%%

w_init = zeros(size(w_true));
opts = optimset('MaxIter',100,'GradObj','on','Hessian','on','display','iter');
w_mle = fminunc(nllFunc_gpu,w_init,opts);


