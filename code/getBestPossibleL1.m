opts = optimset('GradObj','on','Hessian','on','MaxIter',100,'Display','iter');
w_mle = fminunc(nllFunc_gpu,zeros(M,1),opts);

%%
lambdas = 0:25:800;
ws = nan(M,length(lambdas));
llp = zeros(length(lambdas),1);
opts2 = optimset('GradObj','on','Hessian','on','MaxIter',100,'Display','off');
for ii = 1:length(lambdas)
    if(mod(ii,10) == 0)
        fprintf('ideal predictive lambda test %d / %d ...\n',ii,length(lambdas));
    end
    %nllFunc_l1 = @(w) nllGLM_l1(w,X,y,[],[],lambdas(ii));
    nllFunc_l1 = @(weights) nllGLM_l1_GPU(weights,nllFunc_gpu,lambdas(ii));
    if(ii == 1)
        w_init = w_mle;
    else
        w_init = ws(:,ii-1);
    end
    ws(:,ii) = fminunc(nllFunc_l1,w_init,opts2);
    llp(ii) = -nllFunc_pred(ws(:,ii));
end



[maxLL_pred, li] = max(llp);
best_lambda = lambdas(li);
w_bestl1 = ws(:,li);
fprintf('done.\n');

%%
nFold = length(X_gpu);

foldSize = ceil(N/nFold);


llp_xv = zeros(length(lambdas),nFold);
ws_xv = nan(M,length(lambdas),nFold);

for ii = 1:length(lambdas)
    if(mod(ii,10) == 0)
        fprintf('xv lambda test %d / %d ...\n',ii,length(lambdas));
    end
    for jj = 1:nFold
%         test = (1:foldSize) + (jj-1)*foldSize;
%         if(jj == length(lambdas))
%             test = test(1):N;
%         end
%         train = 1:N;
%         train = train(~ismember(train,test));
        test = tts{jj};
        trainBlocks = 1:nFold;
        trainBlocks = trainBlocks(~ismember(trainBlocks,jj));
        nllFunc_gpu2 = @(weights) kcGlmNLL_Multi(weights,1,0,X_gpu(trainBlocks),Y_gpu(trainBlocks),C_gpu(trainBlocks));
        

        %nllFunc_l1 = @(w) nllGLM_l1(w,X(train,:),y(train),[],[],lambdas(ii));
        nllFunc_l1 = @(weights) nllGLM_l1_GPU(weights,nllFunc_gpu2,lambdas(ii));
        if(ii == 1)
            w_init = w_mle;
        else
            w_init = ws_xv(:,ii-1,jj);
        end
        ws_xv(:,ii,jj) = fminunc(nllFunc_l1,w_init,opts2);
        
        nllFunc_gpu3 =  @(w)kcGlmNLL_Multi(w,1,0,X_gpu{jj},Y_gpu{jj},C_gpu{jj});
        llp_xv(ii,jj) = -nllFunc_gpu3(ws_xv(:,ii,jj));
    end
end


[~, li] = max(sum(llp_xv,2));
xv_lambda = lambdas(li);
xvLL_pred = -nllFunc_pred(ws_xv(:,li));

%%
nllFunc_l1 = @(w) nllGLM_l1_GPU(w,nllFunc_gpu,xv_lambda);
w_xvl1 = fminunc(nllFunc_l1,w_mle,opts2);
fprintf('done.\n');

%%

figure(2);
clf
subplot(2,1,1)
plot(lambdas,llp);
subplot(2,1,2)
plot(lambdas,sum(llp_xv,2));
