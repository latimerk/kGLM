function [f,g,h] = glmNll_BerSoftPow(w,X,Y,p,dt)

if(nargin < 5)
    dt = 1.0;
end

if(nargin < 4 || p == 1)
    if(nargout == 1)
        f = glmNll_BerSoftRec(w,X,Y,dt);
    elseif(nargout == 2)
        [f,g] = glmNll_BerSoftRec(w,X,Y,dt);
    else
        [f,g,h] = glmNll_BerSoftRec(w,X,Y,dt);
    end
    return;
end

if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
    LOG_MIN = single(1.1755e-38);
    SPEC_MIN = single(1.1921e-06);
    MIN_POS = single(1e-45);
else
    EXP_MAX = 709.78;
    LOG_MIN = 2.2251e-308;
    SPEC_MIN = 2.2204e-15;
    MIN_POS = 5.0e-324;
end


Xw = X*w;

gt = Xw > 35;
eX = exp(min(EXP_MAX,Xw));

l = Xw;
l(~gt) = max(MIN_POS, log1p(eX(~gt)));
r = l.^p*dt;

y1 = Y>=1;
r_lt = r<1;
a = -r;
a(y1 &  r_lt) = log(max(LOG_MIN,-expm1(-r(y1 & r_lt))));
a(y1 & ~r_lt) = log1p(-exp(min(EXP_MAX,-r(y1 & ~r_lt))));


f = -double(sum(a));

if(nargout > 1)
    nex = 1./(1+exp(min(EXP_MAX,-Xw)));
    inv_dr = l.^(p-1);
    
    r1 = r(y1);
    nex1 = nex(y1);
    inv_dr1 = inv_dr(y1);
    
    exprm1 = expm1(r1);
    
    b1 = (1-nex1)*p;
    b1(exprm1 > SPEC_MIN) = dt*p*inv_dr1(exprm1 > SPEC_MIN)*nex1(exprm1 > SPEC_MIN)/exprm1(exprm1>SPEC_MIN);
    
    b = -dt*nex*p*inv_dr;
    b(y1) = b1;
    
    g = -double(X'*b);
    if(nargout > 2)
        expr1 = exp(min(EXP_MAX,r1));
        c1 =  (b1.^2.*expr1);
        
        c = b.*((1-nex)+nex.*(p-1)./l);
        c(y1) = c(y1)-c1;
        
        if(p >= 1)
            C = X.*sqrt(max(0,-c));
            h = double(C'*C);
        else
            h = double(X'*(X.*(-c)));
        end
    end
end