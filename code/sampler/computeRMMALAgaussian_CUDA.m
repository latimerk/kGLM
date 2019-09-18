function [ll,mu,G,Ginv_chol,mu_base] = computeRMMALAgaussian_CUDA(w_curr,X_gpu,Y_gpu,C_gpu,e,s2,isInv,prior_mu_curr,Z_gpu)

s2_inv = zeros(length(w_curr),length(w_curr));
useS2 = false;
if(nargin > 5 && ~isempty(s2))
    if(numel(s2) == 1)
        s2 = s2*eye(length(w_curr));
    elseif(min(size(s2)) == 1)
        s2 = diag(s2);
    end
    
    
    if(nargin > 6 && ~isempty(isInv) && isInv)
        s2_inv = s2;
    else
        isInv = false;
        s2_inv = inv(s2);
        
    end
    useS2 = true;
end

if(nargin < 8 || isempty(prior_mu_curr))
    prior_mu_curr = zeros(length(w_curr),1);
end
if(nargin < 9)
    Z_gpu = 0;
end

if(iscell(X_gpu) || isstruct(X_gpu))
    [ll,dl,G,H,H2] =  kcGlmRMMALA_Multi(w_curr,s2_inv,X_gpu,Y_gpu,C_gpu,Z_gpu);
else
    error('invalid GPU structs');
end

if(useS2)
    if(isInv)
        ll = ll-1/2*((w_curr-prior_mu_curr)'*(s2_inv*(w_curr-prior_mu_curr)));
        dl = dl-(s2_inv*(w_curr-prior_mu_curr));
    else
        ll = ll-1/2*((w_curr-prior_mu_curr)'*(s2\(w_curr-prior_mu_curr)));
        dl = dl-(s2\(w_curr-prior_mu_curr));
    end
end

%Ginv_X  = G\X';
Ginv = inv(G);
Ginv_chol = chol(Ginv);

mu_base = w_curr  + e^2/2*(G\dl);
mu = mu_base - e^2*sum(G\H,2); %e^2*sum(Ginv_X*eXbX_Ginv_X,2); 
% mu = mu + e^2/2*sum((Ginv_X'.*eXbX)'*X,2);
mu = mu + e^2/2*sum(Ginv,2).*H2; %e^2/2*sum(inv(G).*(eXbX_Ginv_X'*X),2);