function [f,g,h] = nllGLM_l1_GPU(B,nllGPU,lambda)

if(nargin < 3)
    lambda = 0;
end



%f = 0;
g = 0;
h = 0;
if(nargout > 2)
    [f,g,h] = nllGPU(B);
elseif(nargout > 1)
    [f,g] = nllGPU(B);
else
    f = nllGPU(B);
end

f = f + lambda*sum(abs(B));
g = g + lambda*sign(B);

f = f;
g = g;
h = h;
