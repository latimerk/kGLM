function [f,g,h] = glmNll_BerExp(w,X,Y,dt)

if(nargin < 4)
    dt = 1.0;
end
if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
    LOG_MIN = single(1.1755e-38);
else
    EXP_MAX = 709.78;
    LOG_MIN = 2.2251e-308;
end

Xw = X*w + log(dt);
eX = exp(min(EXP_MAX,Xw));

y1 = Y>=1;
r_lt = eX<1;
a = -eX;
a(y1 &  r_lt) = log(max(LOG_MIN,-expm1(-eX(y1 & r_lt))));
a(y1 & ~r_lt) = log1p(-exp(min(EXP_MAX,-eX(y1 & ~r_lt))));


f = -double(sum(a));

if(nargout > 1)
    b = -eX;
    b(y1) = exp(min(EXP_MAX, -eX(y1) - a(y1) + Xw(y1)));
    
    g = -double(X'*b);
    if(nargout > 2)
        c = -eX;
        c(y1) = b(y1).*(-eX(y1) - b(y1) + 1);
        
        C = X.*sqrt(max(0,-c));
        h = double(C'*C);
    end
end