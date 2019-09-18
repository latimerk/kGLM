function [f,g,h] = nllGLM_l1(B,X,y,dt,GLMtype,lambda)

if(nargin < 5 || isempty(GLMtype))
    GLMtype = 0;
end
if(nargin < 4 || isempty(dt))
    dt = 1;
end
if(nargin < 6)
    lambda = 0;
end

Xb = X*B; 
eXb = exp(Xb); 
g = 0;
h = 0;

if(GLMtype == 4)
    r = log(1+eXb);
    f = -sum(r)*dt + y'*log(r);
    if(nargout > 1)
        neXb = exp(-Xb); 
        dr = 1./(1+neXb);
        g = X'*(dr.*(-dt+y./r)); 
        if(nargout > 2)
            d2r = -1./(2.0+neXb+eXb);
            h = X'*bsxfun(@times,X,d2r.*(-dt+y./r) - y.*(dr./r).^2);
        end
    end
else
    f = sum(-eXb) + y'*Xb; 
    if(nargout > 1)
        g = X'*(-eXb+y); 
        if(nargout > 2)
            h = -X'*bsxfun(@times,X,eXb);
        end
    end
end
f = f - lambda*sum(abs(B));
g = g - lambda*sign(B);

f = -f;
g = -g;
h = -h;
