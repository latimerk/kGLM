function [w,log_py] = evidenceApprox(X,Y,lambda)

N = size(X,1);
D = size(X,2);

Xy = X'*Y;
normalizer = (gammaln(Y+1));

opts = optimoptions('fminunc','algorithm','trust-region','gradobj','on','hessian','on','maxiter',200,'display','off');
nllAll   = @(w)glmNll_PoissonExp(w,X,Y,1,Xy,sum(normalizer));
npostAll = @(w)nllGLM_l2_GPU(w,nllAll,lambda);
w = fminunc(npostAll,zeros(D,1),opts);

[f,~,W_inv] = npostAll(w);

log_py = -f + 1/2*logdet(lambda) - 1/2*logdet(W_inv);
