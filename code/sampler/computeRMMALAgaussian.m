function [ll,mu,G,Ginv_chol,mu_base] = computeRMMALAgaussian(w_curr,X,y,Xy,e,s2,s2_2)

Xb = X*w_curr;
eXb  = exp(Xb);
eXbX = bsxfun(@times,eXb,X);

ll = -sum(eXb) + y'*Xb;
dl = -sum(eXbX,1)'+Xy; %-X'*eXb+Xy-(s2.\w);
G  = X'*eXbX;

if(nargin > 5 && ~isempty(s2))
    if(numel(s2) == 1)
        s2 = s2*eye(length(w_curr));
    elseif(min(size(s2)) == 1)
        s2 = diag(s2);
    end
    ll = ll-1/2*(w_curr'*(s2\w_curr));
    dl = dl-(s2\w_curr);
    
    if(nargin < 7)
        s2_2 = s2;
    end
    G  = G+inv(s2_2);
end
M = (eXbX.*(X/G))'*X;%(eXbX_Ginv_X'*X);

%Ginv_X  = G\X';
Ginv = inv(G);
Ginv_chol = chol(Ginv);

%eXbX_Ginv_X = eXbX.*Ginv_X';


mu_base = w_curr  + e^2/2*(G\dl);
mu = mu_base - e^2*sum(G\M,2); %e^2*sum(Ginv_X*eXbX_Ginv_X,2); 
% mu = mu + e^2/2*sum((Ginv_X'.*eXbX)'*X,2);
mu = mu + e^2/2*sum(Ginv.*M,2); %e^2/2*sum(inv(G).*(eXbX_Ginv_X'*X),2);