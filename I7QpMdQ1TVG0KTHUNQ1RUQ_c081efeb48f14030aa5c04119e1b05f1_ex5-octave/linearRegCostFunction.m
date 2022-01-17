function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% size(X)         % 12 x 2
% size(y)         % 12 x 1
% size(theta)     % 2 x 1
% size(lambda)    % 1 x 1

hyp = X * theta;                     % 12 x 1
sumSqDiffs = sum((hyp - y) .^ 2);    % 1 x 1
J = (1/(2*m)) * sumSqDiffs;          % 1 x 1

%drop theta0
theta(1) = 0;
regSum = sum(theta .^ 2);                % 1 x 1
regTerm = (lambda / (2 * m)) * regSum;   % 1 x 1

J = J + regTerm;                         % 1 x 1

grad = (1/m) * ((hyp .- y)' * X);        % basic grad term 1 x 2
grad = grad + (lambda/m) .* theta';       % adding reg term 1 x 2

% =========================================================================

grad = grad(:);

end
