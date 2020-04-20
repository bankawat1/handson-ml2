function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

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

thetaEx = theta
thetaEx(1) = 0

J =  1/m *(sum(-y.*log(hThetaX) - (1-y).*(log(1- hThetaX)))) + (lambda/(2*m))*(sum(thetaEx.^2));

#theta(1) = 0
#gl = ((lambda/m)*theta);
grad = 1/m * (X'*(hThetaX - y)) + ((lambda/m)*thetaEx)




% =============================================================

end
