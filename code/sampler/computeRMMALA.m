function [ll,mu,G,Ginv_chol,mu_base] = computeRMMALA(w,X,y,Xy,e)

Xb = X*w;
eXb  = exp(Xb);
eXbX = bsxfun(@times,eXb,X);

ll = -sum(eXb) + y'*Xb;
dl = -X'*eXb+Xy;
G  = X'*eXbX;
Ginv_X  = G\X';
Ginv = inv(G);
Ginv_chol = chol(Ginv);


mu_base = w  + e^2/2*(G\dl);
mu = mu_base - e^2*sum(Ginv_X*(eXbX.*Ginv_X'),2); 
% mu = mu + e^2/2*sum((Ginv_X'.*eXbX)'*X,2);
mu = mu + e^2/2*sum(inv(G).*((Ginv_X'.*eXbX)'*X),2);