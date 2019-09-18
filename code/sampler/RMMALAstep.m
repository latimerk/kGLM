function [w_curr,accept,lp_accept] = RMMALAstep(w_curr,X_gpu,Y_gpu,C_gpu,Xp_gpu,Xb_gpu,e,s2inv,M_max)
%M_max is max dimensions to sample at one time (chooses random sets of
%dimensions to sample at each time)
%
% if M_max is not given, samples all dimensions. Xp_gpu and Xb_gpu are not
%                                                needed in this case.
%           

M = double(X_gpu{1}.size(2));

if(nargin < 9)
    M_max = M; 
end

N_GPU = length(X_gpu);

if(M < 0 || M_max >= M)
    N = 1;
else
    N = ceil(M/M_max);
end

order = randperm(M);

accept = false(N,1);
lp_accept = zeros(N,1);

Xp_size = zeros(N_GPU,1);
if(N > 1)
    for jj = 1:N_GPU
        Xp_size(jj) = Xp_gpu{jj}.size(2);
    end
end


for ii = 1:N 
    
    
    if(N == 1)
        ws = 1:M;
        Xp_gpu = X_gpu;
        Xb_gpu = 0;
    else
        ws = sort( order(((ii-1)*M_max+1):min(M,ii*M_max)) );
        w_0 = w_curr;
        w_0(ws) = 0;
        
        for jj = 1:N_GPU
            kcMtimesVector(X_gpu{jj},w_0,0,Xb_gpu{jj}); %Xb is a column of constant values in linear term for GLM
            kcArrayCopyColumns(X_gpu{jj},Xp_gpu{jj},int32(ws-1));
            Xp_gpu{jj}.size(2) = length(ws);
        end
        
    end
    nws = 1:M;
    nws = nws(~ismember(nws,ws));
    
    
    b_curr = w_curr(ws);
    
    s2inv_curr = s2inv(ws,ws);
    if(N > 1)
        prior_mu_curr    = -s2inv(ws,ws)\(s2inv(ws,nws)*w_curr(nws));
    else
        prior_mu_curr    = zeros(size(w_curr));
    end
    
    [ll_curr,mu_curr,G_curr,Ginv_chol_curr] = computeRMMALAgaussian_CUDA(b_curr,Xp_gpu,Y_gpu,C_gpu,e,s2inv_curr,true,prior_mu_curr,Xb_gpu);
    b_star = mu_curr + e*Ginv_chol_curr'*randn(size(mu_curr)); %transpose or no?
    %w_star = mvnrnd(mu_curr,e^2*inv(G_curr))';

    %[ll_star,mu_star,G_star,Ginv_chol_star,mu_base_star] = computeRMMALAgaussian(w_star,X,y,Xy,e,s2_curr);
    [ll_star,mu_star,G_star,Ginv_chol_star] = computeRMMALAgaussian_CUDA(b_star,Xp_gpu,Y_gpu,C_gpu,e,s2inv_curr,true,prior_mu_curr,Xb_gpu);

    q_star = -sum(log(diag(Ginv_chol_curr)))-1/(2)*((b_star-mu_curr)'*((e^-2*G_curr)*(b_star-mu_curr)));

    q_curr = -sum(log(diag(Ginv_chol_star)))-1/(2)*((b_curr-mu_star)'*((e^-2*G_star)*(b_curr-mu_star)));


    lp_accept(ii) = ll_star - ll_curr + q_curr - q_star;

    if(log(rand) < lp_accept(ii))
        accept(ii) = true;
        w_curr(ws) = b_star;
    else
        accept(ii) = false;
        w_curr(ws) = b_curr;
    end
    
end

if(N>1)
    for jj = 1:N_GPU
        Xp_gpu{jj}.size(2) = Xp_size(jj);
    end
end