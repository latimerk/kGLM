function [ll_prior,dl_prior,d2l_prior] = logPriorGaussian(w,s2_0,s2_1,s)

w_2 = w(s==2);
w_1 = w(s==1);
w_0 = w(s==0);
dl_prior = zeros(length(w),1);
d2l_prior = zeros(length(w),1);
ll_prior = 0;

p0 = length(w_0);
p1 = length(w_1);
p2 = length(w_2);

if(p0 > 0)
    ll_prior = ll_prior - p0/2*log(2*pi*s2_0)-1/(2*s2_0)*(w_0'*w_0);
    dl_prior(s==0)  = -1/s2_0*w_0;
    d2l_prior(s==0) = -1/s2_0;
end
if(p1 > 0)
    
    ll_prior = ll_prior - p1/2*log(2*pi*s2_1)-1/(2*s2_1)*(w_1'*w_1);
    dl_prior(s==1)  = -1/s2_1*w_1;
    d2l_prior(s==1) = -1/s2_1;
end
if(p2 > 0)
    s2_2 = 100;
    ll_prior = ll_prior - p2/2*log(2*pi*s2_2)-1/(2*s2_2)*(w_2'*w_2);
    dl_prior(s==2)  = -1/s2_2*w_2;
    d2l_prior(s==2) = -1/s2_2;
end

