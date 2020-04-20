function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
alpha = 0.001

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
a = @(x) 1/(1+exp(-x));

hThetaX = arrayfun(a,(theta'*X'))';
#disp(X);
#disp('hThetaX')
#disp(hThetaX)
#firstPart = y.*log(hThetaX);
#disp('firstPart')
#disp(firstPart)

#logMinusOne = log(1-hThetaX)
#disp('logMinusOne')
#disp(logMinusOne)
#secondPart = (1-y).*(logMinusOne);
#disp('secondPart')
#disp(secondPart)
#sumedResult = sum(firstPart + secondPart);
#disp('sumedResult')
#disp(sumedResult)
#J = -1/m*sumedResult;

J =  1/m *(sum(-y.*log(hThetaX) - (1-y).*(log(1- hThetaX))));

grad = 1/m * (X'*(hThetaX - y))




% =============================================================

end
