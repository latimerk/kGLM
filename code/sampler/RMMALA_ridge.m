
NS = 1000;

Xy = X'*y;

lp_accept = nan(NS,1);
accept = nan(NS,1);
lls = nan(NS,1);

w = nan(M,NS);
ps0_w = nan(M,NS);
w(:,1) = w_mle;


s2 = 0.1^2;

e = 0.7;



ll_pred = nan(NS,1);
ll_pred_mean = nan(NS,1);
ll_pred_median = nan(NS,1);
w_mean  = nan(M,NS);

ll_pred_mle = -nllFunc_pred(w_mle);
ll_pred_true = -nllFunc_pred(w_true);


IG = makedist('InverseGaussian');

w_curr = w(:,1);
s2_curr = s2;
[ll_curr,mu_curr,G_curr,Ginv_chol_curr,mu_base_curr] = computeRMMALAgaussian_CUDA(w_curr,X_gpu,Y_gpu,C_gpu,e,s2_curr);

for ss = 2:NS
    %w_curr = w(:,ss-1);
    s2_curr = s2;

    %[ll_curr,mu_curr,G_curr,Ginv_chol_curr,mu_base_curr] = computeRMMALAgaussian(w_curr,X,y,Xy,e,s2_curr);
    %[ll_curr,mu_curr,G_curr,Ginv_chol_curr,mu_base_curr] = computeRMMALAgaussian_CUDA(w_curr,X_gpu,Y_gpu,C_gpu,e,s2_curr);

    w_star = mu_curr + e*Ginv_chol_curr'*randn(size(mu_curr)); %transpose or no?
    %w_star = mvnrnd(mu_curr,e^2*inv(G_curr))';

%     [ll_star,mu_star,G_star,Ginv_chol_star,mu_base_star] = computeRMMALAgaussian(w_star,X,y,Xy,e,s2_curr);
    [ll_star,mu_star,G_star,Ginv_chol_star,mu_base_star] = computeRMMALAgaussian_CUDA(w_star,X_gpu,Y_gpu,C_gpu,e,s2_curr);

    q_star = -sum(log(diag(Ginv_chol_curr)))-1/(2)*((w_star-mu_curr)'*((e^-2*G_curr)*(w_star-mu_curr)));

    q_curr = -sum(log(diag(Ginv_chol_star)))-1/(2)*((w_curr-mu_star)'*((e^-2*G_star)*(w_curr-mu_star)));


    lp_accept(ss) = ll_star - ll_curr + q_curr - q_star;

    if(log(rand) < lp_accept(ss))
        accept(ss) = true;
        w(:,ss) = w_star;
        lls(ss) = ll_star;
        
        ll_curr = ll_star;
        mu_curr = mu_star;
        G_curr  = G_star;
        Ginv_chol_curr = Ginv_chol_star;
        mu_base_curr   = mu_base_star;
        w_curr = w_star;
    else
        accept(ss) = false;
        w(:,ss) = w_curr;
        lls(ss) = ll_curr;
    end
    
    
    
    
    %%
    s_start = 500;
    if(ss-500 < 600)
        s_start = 2;
    end
    w_mean(:,ss) = nanmean(w(:,s_start:ss),2);
    ll_pred_mean(ss) = -nllFunc_pred(w_mean(:,ss));
    ll_pred_median(ss) = -nllFunc_pred(nanmedian(w(:,s_start:ss),2));
    
    
    %%
    if(mod(ss,10) == 0)
        sfigure(2); clf
        subplot(3,2,1);
         hold on
        idx = 1:min(15,M);
        
        
        plot(w(idx(1),:),w(idx(2),:))
        plot(w_true(idx(1)),w_true(idx(2)),'rx');
        plot(w_mle(idx(1)),w_mle(idx(2)),'kx');
        plot(w_mean(idx(1),ss),w_mean(idx(2),ss),'gx');
        if(exist('w_xvl1','var'))
            plot(w_xvl1(idx(1)),w_xvl1(idx(2)),'mx');
        end
        if(exist('w_bestl1','var'))
            plot(w_bestl1(idx(1)),w_bestl1(idx(2)),'cx');
        end
        title(sprintf('ss = %d / %d, accept rate = %2.1f',ss,NS,nanmean(accept)*100));
        hold off
        
        
        subplot(3,2,3);
        hold on
        plot(1:NS,lls);
        plot([1 NS], -nllFunc_cpu(w_mle)*[1 1],'k--');
        plot([1 NS], -nllFunc_cpu(w_true)*[1 1],'r--');
        title('train LL');
        hold off
        
        
        subplot(3,2,5);
        hold on
        plot(1:NS,ll_pred_mean,'b');
        plot(1:NS,ll_pred_median,'b--');
        plot([1 NS], ll_pred_mle*[1 1],'k--');
        plot([1 NS], ll_pred_true*[1 1],'r--');
        if(exist('maxLL_pred','var'))
            plot([1 NS], [maxLL_pred maxLL_pred],'c--');
        end
        if(exist('xvLL_pred','var'))
            plot([1 NS], [xvLL_pred xvLL_pred],'m--');
        end
        title('predictive LL');
        hold off
        
        for ii = 1:length(idx)
            subplot(length(idx),2,ii*2);
            hold on
            plot(1:NS,w(idx(ii),:));
            plot([1 NS],[1 1]*w_mle(idx(ii)),'--k');
            plot([1 NS],[1 1]*w_true(idx(ii)),'--r');
            
            hold off
            
        end
        
        drawnow;
    end
end