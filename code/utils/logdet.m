function logdetA = logdet(G)

try
    G_det = chol(G);
    logdetA = 2*sum(log(diag(G_det)));
catch
%     [~, U, P] = lu(h);
%     du = diag(U);
%     c = det(P) * prod(sign(du));
%     logdetA = log(c) + sum(log(abs(du)));
    %logdetA = log(det(G));
    try
        [~,s,~] = svd(G);
        logdetA = sum(log(diag(s)));
    catch
        logdetA = log(max(eps,det(G)));
    end
end