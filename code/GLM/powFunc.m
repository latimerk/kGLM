function [Y] = powFunc(X,p)

Y = zeros(size(X));

g_0 = 1;
g_1 = p;
g_2 = p*(p-1);

a = g_0;
c = g_1/a;
b = 2*a*c^2-g_2;

d = 0;
if(p ~= 1)
    d = g_2/(p*(p-1));
end
e = g_1 - p*d;
f = g_0 - d;

X_n = X(X <= 0);
X_p = X(X >  0);

Y(X <= 0) = a./(b*X_n.^2 - c*X_n + 1);
Y(X >  0) = d*(X_p+1).^p + e*X_p + f;