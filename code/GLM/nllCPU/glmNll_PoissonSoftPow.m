function [f,g,h] = glmNll_PoissonSoftPow(w,X,Y,p,dt,normalizer)

if(nargin < 5)
    dt = 1.0;
end
if(nargin < 6)
    normalizer = sum(Y)*log(dt)-sum(gammaln(Y+1));
end

if(nargin < 4 || p == 1)
    if(nargout == 1)
        f = glmNll_PoissonSoftRec(w,X,Y,dt,normalizer);
    elseif(nargout == 2)
        [f,g] = glmNll_PoissonSoftRec(w,X,Y,dt,normalizer);
    else
        [f,g,h] = glmNll_PoissonSoftRec(w,X,Y,dt,normalizer);
    end
    return;
end

if(p <= 0)
    error("Invalid power - must be positive.");
end

if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
    LOG_MIN = single(1.1755e-38);
    MIN_POS = single(1e-45);
else
    EXP_MAX = 709.78;
    LOG_MIN = 2.2251e-308;
    MIN_POS = 5.0e-324;
end

Xw = X*w;
eX = exp(min(EXP_MAX,Xw));
a  = Xw;
lt = Xw < 35;
a(lt) = max(MIN_POS,log1p(eX(lt)));
r  = a.^p;

f = double(-Y'*p*log(max(LOG_MIN,a)) + sum(r)*dt - normalizer);

if(nargout > 1)
    nex = exp(min(EXP_MAX,-Xw));
    inv_dr = (1+nex).*a.^(1-p)./p;
    
    gt = Xw > -35;
    inv_dr_r = Xw;
    inv_dr_r(~gt) = 1;
    inv_dr_r(lt & gt) = (1+nex(lt & gt)).*a(lt & gt);
    inv_dr_r = inv_dr_r./p;
    
    g = double(X'*(dt./inv_dr - Y./inv_dr_r));
    if(nargout > 2)
        b = 1 ./(1 + eX) + (p-1)./(p*inv_dr_r);
        inv_d2r = inv_dr./b;
        inv_d2r_r = inv_dr_r./b;
        d2 = Y./(inv_dr_r).^2 + (dt./inv_d2r - Y./inv_d2r_r);
        
        if(p >= 1)
            C = X.*sqrt(max(0,d2));
            h = double(C'*C);
        else
            h = double(X'*(X.*d2));
        end
    end
end