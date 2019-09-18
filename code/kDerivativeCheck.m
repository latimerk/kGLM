function [f,g,h,g_est,h_est,h_est2] = kDerivativeCheck(func,point,dx)
%f,g,h are the function evaluations at that point
%g_est is a finite difference estimate of the derivative
%h_est, h_est2 are different estimates of the hessian
%       h_est2 uses the gradient from the given function to 
%              predict the hessian - if the gradient is wrong, 
%              this'll really be wrong


if(nargin < 3)
    dx = 1e-4;
end

[f,g,h] = func(point);

c  = [ 1/280 -4/105  1/5 -4/5    0    4/5 -1/5 4/105 -1/280]./dx;
c2 = [-1/560  8/315 -1/5  8/5 -205/72 8/5 -1/5 8/315 -1/560]./dx^2;

xx  = (-4:4)*dx;
K = length(point);

g_est  = zeros(K,1);
h_est  = zeros(K,K);
h_est2 = zeros(K,K);

for ii = 1:K
    fprintf('k = %d / %d\n',ii,K);
    
    fs = zeros(length(c),1);
    gs = zeros(length(c),K);
    for nn = 1:length(c)
        x = point;
        x(ii) = x(ii) + xx(nn);
        
        if(nargout > 5)
            [fs(nn),gc] = func(x);
            gs(nn,:) = gc;
        else
            fs(nn) = func(x);
        end
    end
    
    g_est(ii)    = c*fs;
    if(nargout > 4)
        h_est(ii,ii) = c2*fs;

        for jj = ii+1:K
            x = point;
            x(ii) = x(ii) + dx;
            x(jj) = x(jj) + dx;
            f_pp = func(x);

            x = point;
            x(ii) = x(ii) + dx;
            x(jj) = x(jj) - dx;
            f_pm = func(x);

            x = point;
            x(ii) = x(ii) - dx;
            x(jj) = x(jj) + dx;
            f_mp = func(x);

            x = point;
            x(ii) = x(ii) - dx;
            x(jj) = x(jj) - dx;
            f_mm = func(x);

            h_est(ii,jj) = (f_pp - f_pm - f_mp + f_mm)/(4*dx^2);
            h_est(jj,ii) = h_est(ii,jj);
        end
    end
    
    if(nargout > 5)
        for jj = ii:K
            h_est2(ii,jj) = c*gs(:,jj);
            h_est2(jj,ii) = h_est2(ii,jj);
        end
    end
end
    
    