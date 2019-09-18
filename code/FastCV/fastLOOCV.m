% experimental, and likely useless, approximations of the GLM LOOCV
function [w,lpe,f,a,b] = fastLOOCV(X,Y,lambda)

N = size(X,1);
D = size(X,2);

Xy = X'*Y;
normalizer = (gammaln(Y+1));

opts = optimoptions('fminunc','algorithm','trust-region','gradobj','on','hessian','on','maxiter',200,'display','off');
nllAll   = @(w)glmNll_PoissonExp(w,X,Y,1,Xy,sum(normalizer));
npostAll = @(w)nllGLM_l2_GPU(w,nllAll,lambda);
w = fminunc(npostAll,zeros(D,1),opts);

[~,~,W_inv] = npostAll(w);

f = nllAll(w);

mu   = X*w;
sig2 = sum((X/W_inv).*X,2);

cc = (exp(sig2)-1);
a = 1./cc;
log_b = -mu-1/2*sig2 - log(cc);
b = exp(log_b);

lpe = nan(N,4);

% lpe(:,1) = -(a-y-1)./b+y.*log((a-y-1)./b);
% lpe(:,1) = -(a-Y-1)./b+Y.*(log(a-Y-1) - log_b) - normalizer;
lpe(:,1) = -(1-Y.*cc-cc)./(cc.*b)+Y.*(log(a-Y-1) - log_b) - normalizer;

% lpe(:,2) = (a-Y).*log(b-1) - gammaln(a-Y) + gammaln(a) - a.*log_b - normalizer;
lpe(:,2) = (a-Y).*log(expm1(log_b)) - gammaln(a-Y) + gammaln(a) - a.*log_b - normalizer;


XwX = sum((X*W_inv).*X,2);

XX = sum(X.*X,2);

c = exp(min(90,mu));
%d = XwX.*exp(min(90,-mu - 2*log(XX)));

log_b = log(XwX) - 2*log(XX);
b = XwX.*exp(min(90,- 2*log(XX)));%d.*c;
a = c.*b + 1;


lpe(:,3) = -(a-Y-1)./b+Y.*(log(a-Y-1) - log_b) - normalizer;
lpe(:,4) = (a-Y).*log(b-1) - gammaln(a-Y) + gammaln(a) - a.*log_b - normalizer;



if(sum(b <= 1) > 0)
    warning('Not all beta values appear stable!');
end
if(sum(a <= Y+1) > 0)
    warning('Not all alpha values appear stable!');
end
