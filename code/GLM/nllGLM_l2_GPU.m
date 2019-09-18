function [f,g,h] = nllGLM_l2_GPU(B,nllGPU,lambda)

if(nargin < 3)
    lambda = 0;
end

%f = 0;
g = 0;
h = 0;
lb = lambda*B;
if(nargout > 2)
    [f,g,h] = nllGPU(B);
    g = g + lb;
    h = h + lambda;
    
    if(sum(isinf(h(:))) > 0 || sum(isnan(h(:))) > 0)
        fprintf('here (nllGLM_l2_GPU) \n');
    end
elseif(nargout > 1)
    [f,g] = nllGPU(B);
    g = g + lb;
else
    f = nllGPU(B);
end

f = f + 1/2*(B'*lb);




