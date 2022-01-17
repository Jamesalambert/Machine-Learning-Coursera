function c = cov(X)
    
    X = X(:);
    n = size(X,2 );
    m = size(X,1);
    c = zeroes(n,n);
    
    c = (1/m) * (X * X');
end