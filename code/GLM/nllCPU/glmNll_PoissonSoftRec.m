function [f,g,h] = glmNll_PoissonSoftRec(w,X,Y,dt,normalizer)

if(nargin < 4)
    dt = 1.0;
end
if(nargin < 5)
    normalizer = sum(Y)*log(dt)-sum(gammaln(Y+1));
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
r  = Xw;
lt = Xw < 35;
r(lt) = max(MIN_POS,log1p(eX(lt)));

f = double(-Y'*log(max(LOG_MIN,r)) + sum(r)*dt - normalizer);

if(nargout > 1)
    nex = exp(min(EXP_MAX,-Xw));
    inv_dr = 1+nex;
    
    gt = Xw > -35;
    inv_dr_r = Xw;
    inv_dr_r(~gt) = 1;
    inv_dr_r(lt & gt) = r(lt & gt).*inv_dr(lt & gt);
    
    g = double(X'*(dt./inv_dr - Y./inv_dr_r));
    if(nargout > 2)
        inv_d2r = (1+eX).*inv_dr;
        inv_d2r_r = inv_dr_r.*(1+eX);
        d2 = Y./(inv_dr_r).^2 + (dt./inv_d2r - Y./inv_d2r_r);
        
        C = X.*sqrt(max(0,d2));
        h = double(C'*C);
    end
end