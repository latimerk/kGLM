function [f,g,h] = glmNll_BerLogit(w,X,Y)


if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
else
    EXP_MAX = 709.78;
end


a = (Y>=1)*-2 +1;
Xw = a.*(X*w);

eX = Xw;
lt = Xw <35;
eX(lt) = log1p(exp(Xw(lt)));

f = double(sum(eX));

if(nargout > 1)
    nex = 1./(1+exp(min(EXP_MAX,-Xw)));
    
    g = double(X'*(a.*nex));
    if(nargout > 2)
        c = nex.*(1-nex);
        
        C = X.*sqrt(max(0,c));
        h = double(C'*C);
    end
end