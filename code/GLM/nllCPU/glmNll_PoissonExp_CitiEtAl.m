
% Likelihood Methods for Point Processes with Refractoriness
% Luca Citi, Demba Ba, Emery N. Brown and Riccardo Barbieri
% Neural Computation 2014

function [f,g,h] = glmNll_PoissonExp_CitiEtAl(w,X,Y,dt,Xy)

if(nargin < 4)
    dt = 1.0;
end
if(nargin < 5)
    Xy = X'*Y;
end

if(isa(X,'single'))
    w = single(w);
    EXP_MAX = single(88.7);
else
    EXP_MAX = 709.78;
end

Xw = min(EXP_MAX,X*w + log(dt));

y2 = log(1-Y./2);
eX = exp(Xw + y2);

f = double(-Xy'*w + sum(eX) );

if(nargout > 1)
    g = double(X'*(eX) - Xy);
    if(nargout > 2)
        C = X.*(exp((Xw + y2)*0.5));
        h = double(C'*C);
%         h = double(X'*(eX));
    end
end