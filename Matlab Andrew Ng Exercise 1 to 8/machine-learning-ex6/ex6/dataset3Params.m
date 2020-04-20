function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_arr = [0.01,0.03,0.1,0.3,1,3,10,30];
sig = [0.01,0.03,0.1,0.3,1,3,10,30];
accuracy = zeros(64,3);
m = 1;

for i = C_arr
  for j = sig
        model = svmTrain(X,y,i,@(x1,x2) gaussianKernel(x1, x2, j));
        predictions = svmPredict(model,Xval);
        accuracy(m,3) = mean(double(predictions == yval));
        accuracy(m,1) = i;
        accuracy(m,2) = j;
        m = m +1;
        disp(m);
  endfor
endfor
disp("accuracy");
disp(accuracy);

disp("max accuracy");
[acc max_accurate] = max(accuracy(:,3));
C = accuracy(max_accurate,1);
sigma = accuracy(max_accurate,2);
fprintf("C: %f\r\n signma: %f\r\n accuracy : %f\r\n",C,sigma,acc);

input("paused");





% =========================================================================

end
