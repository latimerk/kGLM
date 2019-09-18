
NS = 10000;

Xy = X'*y;

lls = nan(NS,1);
lp_accept = nan(NS,ceil(M/M_max));
accept = nan(NS,ceil(M/M_max));

w = nan(M,NS);
ps0_w = nan(M,NS);
w(:,1) = w_mle;


e = 0.7;

tau = nan(M,NS);
lambda = nan(NS,1);

tau(:,1) = 0.01^2;
lambda(1) = 100;


ll_pred_mean = nan(NS,1);
ll_pred_median = nan(NS,1);
w_mean  = nan(M,NS);

ll_pred_mle = -nllFunc_pred(w_mle);
ll_pred_true = -nllFunc_pred(w_true);

lambda_prior_shape = 0.5;
lambda_prior_rate  = 0.00001;


IG = makedist('InverseGaussian');

%%
for ss = 2:NS
    w_curr  = w(:,ss-1);
    %s2_curr = tau(:,ss-1);
    s2_inv = diag(1./tau(:,ss-1));


    
    [w(:,ss),accept(ss,:),lp_accept(ss,:)] = RMMALAstep(w_curr,X_gpu,Y_gpu,C_gpu,XP_gpu,Xb_gpu,e,s2_inv,M_max);
    
    
    
    %% sample \tau
    for jj = 1:M
        IG.mu      = sqrt(lambda(ss-1)^2/w(jj,ss)^2);
        IG.lambda  = lambda(ss-1)^2;
        tau(jj,ss) = min(1e10,max(1e-20,1/IG.random()));
    end
    
    %% sample \lambda
    if(mod(ss,1) == 0)
        lambda_post_shape = M                + lambda_prior_shape;
        lambda_post_rate  = sum(tau(:,ss))/2 + lambda_prior_rate;
        lambda(ss) = sqrt(gamrnd(lambda_post_shape, 1/lambda_post_rate));
    else
        lambda(ss) = lambda(ss-1);
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
        sfigure(3); clf
        subplot(4,2,1);
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
        title(sprintf('ss = %d / %d, accept rate = %2.1f',ss,NS,nanmean(accept(:))*100));
        hold off
        
        subplot(4,2,3);
        hold on
        plot(1:NS,lambda);
        if(exist('best_lambda','var'))
            plot([1 NS], [best_lambda best_lambda],'c--');
        end
        if(exist('xv_lambda','var'))
            plot([1 NS], [xv_lambda xv_lambda],'m--');
        end
        xlim([0 NS]);
        hold off
        
        %this plot doesn't really work
%         subplot(4,2,5);
%         hold on
%         plot(1:NS,lls);
%         plot([1 NS], -nllFunc_cpu(w_mle)*[1 1],'k--');
%         plot([1 NS], -nllFunc_cpu(w_true)*[1 1],'r--');
%         title('train LL - this');
%         hold off
        
        
        subplot(4,2,7);
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