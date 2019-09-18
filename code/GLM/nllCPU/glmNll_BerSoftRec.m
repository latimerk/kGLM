function [f,g,h] = glmNll_BerSoftRec(w,X,Y,dt)

if(nargin < 4)
    dt = 1.0;
end
if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
    LOG_MIN = single(1.1755e-38);
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

r = Xw;
r(~gt) = max(MIN_POS,log1p(eX(~gt)));
r = r*dt;

y1 = Y>=1;
r_lt = r<1;
a = -r;
a(y1 &  r_lt) = log(max(LOG_MIN,-expm1(-r(y1 & r_lt))));
a(y1 & ~r_lt) = log1p(-exp(min(EXP_MAX,-r(y1 & ~r_lt))));


f = -double(sum(a));

if(nargout > 1)
    nex = 1./(1+exp(min(EXP_MAX,-Xw)));
    
    r1 = r(y1);
    nex1 = nex(y1);
    expr = (1+eX(y1)).^dt;
    exprm1 = r1;
    exprm1(r1>0.01)  = expr(r1>0.01) - 1;
    exprm1(r1<=0.01) = expm1(r1(r1<=0.01)); 
    
    b1 = dt*(1-nex1);
    b1(exprm1 > SPEC_MIN) = dt*nex1(exprm1 > SPEC_MIN)./exprm1(exprm1>SPEC_MIN);
    
    b = -dt*nex;
    b(y1) = b1;
    
    g = -double(X'*b);
    if(nargout > 2)
        c1 = r1;
        c1(exprm1 >  SPEC_MIN) = dt*(1-nex1(exprm1 >  SPEC_MIN)).*nex1(exprm1 >  SPEC_MIN)./exprm1(exprm1 >  SPEC_MIN) - expr(exprm1 >  SPEC_MIN).*b1(exprm1 >  SPEC_MIN).^2;
        c1(exprm1 <= SPEC_MIN) = dt*(1-nex1(exprm1 <= SPEC_MIN)).^2                                                    - expr(exprm1 <= SPEC_MIN).*b1(exprm1 <= SPEC_MIN).^2;
        
        c = -dt*nex.*(1-nex);
        c(y1) = c1;
        
        C = X.*sqrt(max(0,-c));
        h = double(C'*C);
    end
end