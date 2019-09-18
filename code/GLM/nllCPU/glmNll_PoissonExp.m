function [f,g,h] = glmNll_PoissonExp(w,X,Y,dt,Xy,normalizer)

if(nargin < 4)
    dt = 1.0;
end
if(nargin < 5)
    Xy = X'*Y;
end
if(nargin < 6)
    normalizer = sum(Y)*log(dt)-sum(gammaln(Y+1));
end
if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
else
    EXP_MAX = 709.78;
end

Xw = min(EXP_MAX,X*w + log(dt));
eX = exp(Xw);

f = double(-Xy'*w + sum(eX) - normalizer);

if(nargout > 1)
    g = double(X'*eX - Xy);
    if(nargout > 2)
        C = X.*exp(Xw*0.5);
        h = double(C'*C);
%         h = double(X'*(eX));
    end
end