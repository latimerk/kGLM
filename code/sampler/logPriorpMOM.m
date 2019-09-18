function [ll_prior,dl_prior,d2l_prior] = logPriorpMOM(w,r,s2_0,s2_1,s)

w_2 = w(s==2);
w_1 = w(s==1);
w_0 = w(s==0);
dl_prior = zeros(length(w),1);
d2l_prior = zeros(length(w),1);
ll_prior = 0;

p = length(w_1);
p2 = length(w_0);
p3 = length(w_2);

if(p > 0)
    w_1(w_1 == 0) = 1e-30; %in the model, this is false a.s. for all w
    
    logdp = -p*(r*log(2) + gammaln(1/2+r) - 1/2*log(pi));
    ll_prior = ll_prior + logdp -1/2*log(2*pi) -(r + 1/2)*log(s2_1) - 1/(2*s2_1)*(w_1'*w_1) + r*sum(log(max(w_1.^2,1e-300)));
    dl_prior(s==1)  = -1/s2_1*w_1 + 2*r./(sign(w_1).*max(abs(w_1),1e-30));
    d2l_prior(s==1) = -1/s2_1 - 2*r./max(w_1.^2,1e-30);
end
if(p2>0)
    ll_prior = ll_prior - p2*log(2*pi*s2_0)-1/(2*s2_0)*(w_0'*w_0);
    dl_prior(s==0)  = -1/s2_0*w_0;
    d2l_prior(s==0) = -1/s2_0;
end
if(p3>0)
    s2_2 = 100;
    ll_prior = ll_prior - p2*log(2*pi*s2_2)-1/(2*s2_2)*(w_2'*w_2);
    dl_prior(s==2)  = -1/s2_2*w_2;
    d2l_prior(s==2) = -1/s2_2;
end


