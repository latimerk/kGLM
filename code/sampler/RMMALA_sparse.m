
NS = 1e3;

Xy = X'*y;

lp_accept = nan(NS,ceil(M/M_max));
accept = nan(NS,ceil(M/M_max));
lls = nan(NS,1);

w = nan(M,NS);
s = nan(M,NS);
ps0_w = nan(M,NS);
w(:,1) = w_mle;
s(:,1) = 1;

phi = nan(NS,1);
phi_a = nan(NS,1);
phi_b = nan(NS,1);
phi(1) = 0.5;

e = 0.7;

s2_0_c = 1000;

use_s2_c = false;
sampleS2 = false;

s2_1 = nan(NS,1);
s2_0 = nan(NS,1);
s2_1(1) = 1^2;%(0.02)^2;
if(use_s2_c)
    s2_0(1)    = s2_1(1)/s2_0_c;
else
    s2_0(1)    = (0.001)^2;
end

phi_prior_a = 1;
phi_prior_b = 1;

s2_prior_a = 1;%5;
s2_prior_b = 0.01;%0.01;

ll_pred_mean = nan(NS,1);
ll_pred_median = nan(NS,1);
w_mean  = nan(M,NS);

ll_pred_mle = -nllFunc_pred(w_mle);
ll_pred_true = -nllFunc_pred(w_true);

%%
for ss = 2:NS
    w_curr = w(:,ss-1);
    s2_curr                 = ones(M,1)*s2_0(ss-1);
    s2_curr(s(:,ss-1) == 1) = s2_1(ss-1);
   
    
    s2_curr = diag(s2_curr);
    s2_inv = inv(s2_curr);

    [w(:,ss),accept(ss,:),lp_accept(ss,:)] = RMMALAstep(w_curr,X_gpu,Y_gpu,C_gpu,XP_gpu,Xb_gpu,e,s2_inv,M_max);
    
    %% sample s
    for ii = 1:M
        lpw_s1 = logPriorGaussian(w(ii,ss),s2_0(ss-1),s2_1(ss-1),1);
        lpw_s0 = logPriorGaussian(w(ii,ss),s2_0(ss-1),s2_1(ss-1),0);
        lps1 = log(phi(ss-1));
        lps0 = log(1-phi(ss-1));

        t0 = lpw_s0+lps0;
        t1 = lpw_s1+lps1;

        c = max(t0,t1) - 8;
        t0 = t0 - c;
        t1 = t1 - c;

        ps0_w(ii,ss) = (exp(t0))/(exp(t0) + exp(t1));
        %ps1_w = 1-ps0_w;

        if(rand < ps0_w(ii,ss))
            s(ii,ss) = 0;
        else
            s(ii,ss) = 1;
        end

    end
    %% sample phi
    phi_a(ss) = phi_prior_a + sum(s(:,ss) == 1);
    phi_b(ss) = phi_prior_b + sum(s(:,ss) == 0);
    phi(ss) = betarnd(phi_a(ss),phi_b(ss));
    %phi(ss) = phi(ss-1);
    
    %% sample s2_1
    if(sampleS2)
        if(~use_s2_c)
            s2_post_a = s2_prior_a + sum(s(:,ss) == 1)/2;
            s2_post_b = s2_prior_b + sum(w(s(:,ss) == 1).^2)/2;
            s2_1(ss) = 1/gamrnd(s2_post_a,1/s2_post_b);
            s2_0(ss) = s2_0(ss-1);
        else
            s2_post_a = s2_prior_a + M/2;
            s2_post_b = s2_prior_b + sum(w(s(:,ss) == 1).^2)/2 + s2_0_c*sum(w(s(:,ss) == 0).^2)/2;
            s2_1(ss) = 1/gamrnd(s2_post_a,1/s2_post_b);
            s2_0(ss) = s2_1(ss)/s2_0_c;
        end
    else
        s2_1(ss) = s2_1(ss-1);
        s2_0(ss) = s2_0(ss-1);
    end
    
    %%
    s_start = 500;
    if(ss-500 < 600)
        s_start = 2;
    end
    w_mean(:,ss) = nanmean(w(:,s_start:ss),2);
    ll_pred_mean(ss)   = -nllFunc_pred(w_mean(:,ss));
    ll_pred_median(ss) = -nllFunc_pred(nanmedian(w(:,s_start:ss),2));
    %%
    if(mod(ss,10) == 0)
        fprintf('ss = %d\n',ss);
        if(sampleS2)
            sfigure(10);
            clf
            if(use_s2_c)
                subplot(2,1,1)
                plot(1:NS,sqrt(s2_1))
                xlim([1 NS])
                subplot(2,1,2)
                plot(1:NS,sqrt(s2_0))
                xlim([1 NS])
            else
                plot(1:NS,sqrt(s2_1))
                xlim([1 NS])
            end
        end
        
        sfigure(1); clf
        subplot(3,3,1);
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
        title(sprintf('ss = %d / %d, accept rate = %2.1f',ss,NS,nanmean(nanmean(accept,2))*100));
        hold off
        
        
        subplot(3,3,4);
        hold on
        plot(1:NS,lls);
        plot([1 NS], -nllFunc_cpu(w_mle)*[1 1],'k--');
        plot([1 NS], -nllFunc_cpu(w_true)*[1 1],'r--');
        title('train LL');
        hold off
        
        
        subplot(3,3,7);
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
            subplot(length(idx),3,ii*3-1);
            hold on
            plot(1:NS,w(idx(ii),:));
            plot([1 NS],[1 1]*w_mle(idx(ii)),'--k');
            plot([1 NS],[1 1]*w_true(idx(ii)),'--r');
            
            hold off
            
            subplot(length(idx),3,ii*3);
            hold on
            plot(1:NS,s(idx(ii),:),'g');
            plot([1 NS],[1 1]*s_true(idx(ii)),'--r');
            plot(1:NS,1-ps0_w(idx(ii),:));
            ylim([-0.1 1.1]);
            hold off
        end
        
        drawnow;
    end
end