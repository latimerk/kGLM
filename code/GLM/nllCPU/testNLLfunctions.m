testType = 6;

p = 2.2;
dt = 0.25;

dx = 0.001;
Ws = -100:dx:100;

N = length(Ws);

f = zeros(N,2);
g = zeros(N,2);
h = zeros(N,2);

w = 1;
for ii = 1:N
    for jj = 1:2
        X = 1;
        Y0 = jj-1;
        switch testType
            case 1
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_PoissonExp(Ws(ii),X,Y0,dt);
            case 2
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_PoissonSoftRec(Ws(ii),X,Y0,dt);
            case 3
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_PoissonSoftPow(Ws(ii),X,Y0,p,dt);
            case 4
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_BerExp(Ws(ii),X,Y0,dt);
            case 5
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_BerSoftRec(Ws(ii),X,Y0,dt);
            case 6
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_BerSoftPow(Ws(ii),X,Y0,p,dt);
            case 7
                [f(ii,jj),g(ii,jj),h(ii,jj)] = glmNll_BerLogit(Ws(ii),X,Y0);
        end
    end
end

%%

g_e1 = diff(f)./dx;
g_e = ([g_e1;g_e1(end,:)] + [g_e1(1,:);g_e1])./2;

h_e1 = diff(g)./dx;
h_e = ([h_e1;h_e1(end,:)] + [h_e1(1,:);h_e1])./2;

figure(1);
clf();
subplot(3,3,1)
plot(Ws,f);
subplot(3,3,2)
plot(Ws,g);
subplot(3,3,3)
plot(Ws,h);

if(testType>3)
    subplot(3,3,4)
    ps = exp(-f);
    tp = sum(ps,2);
    plot(Ws,[ps tp]);
end
    

subplot(3,3,5)
plot(Ws(2:end-1),[g(2:end-1,1)-g_e(2:end-1,1)]);
subplot(3,3,8)
plot(Ws(2:end-1),[g(2:end-1,2)-g_e(2:end-1,2)]);

subplot(3,3,6)
plot(Ws(2:end-1),[h(2:end-1,1)-h_e(2:end-1,1)]);
subplot(3,3,9)
plot(Ws(2:end-1),[h(2:end-1,2)-h_e(2:end-1,2)]);