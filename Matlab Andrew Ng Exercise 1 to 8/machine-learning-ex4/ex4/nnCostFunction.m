function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%Feedforwarding ...
X = [ones(size(X,1),1) X];

z2 = Theta1*X'; %Theta1 is 25 X 401... X' is 401 X 5000.... z2 is 25 X 5000

a2 = sigmoid(z2); 

a2 = [ones(1, size(a2,2));a2]; % a2 is 26 X 5000
z3 = Theta2 * a2; % Theta2 is 10 X 26 ... z3 is 10 X 5000

a3 = sigmoid(z3);

yVec = zeros(num_labels,size(y,1));

disp(size(y));
for i = 1:size(y,1)
  r = y(i);
  yVec(r,i) = 1;
  
endfor

J = 1/m *(sum((sum(-yVec.*log(a3) - (1-yVec).*(log(1- a3))))));

J = J + (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))+ sum(sum(Theta2(:,2:end).^2))));

%input"Cost calculated");
%disp(Theta1);
%disp(size(Theta1));
%disp(size(Theta1(:,2:end)))
%a =  Theta1(:,2:end);
%disp(Theta1(:,1));
%disp(a(:,1));
%fprintf("lambda %d", lambda);
%sum1 = sum(sum(Theta1(:,2:end).^2));
%sum2 =sum(sum(Theta2(:,2:end).^2));
%l =lambda/(2*m);
%disp(l* (sum1 +sum2));

%disp(Theta1(:,2:end).^2)
%disp(J);
%input("J calculated");

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%input"starting BackPropogation");

s_delta3 = a3 - yVec; %it is 10 X 5000 ...


s_delta2 = Theta2'*s_delta3.* (a2.*(1-a2));  %Theta2' is 26 X 10 and s_delta3 is 10 X 5000, a2 is 26 X 5000
s_delta2 = s_delta2(2:end,:); % that is 25 X 5000...

big_delta2 = zeros(size(Theta2));
big_delta1 = zeros(size(Theta1));

%input"Starting big_delta2");
for i=1:size(X,1)
  big_delta2 = big_delta2 + s_delta3(:,i)* a2(:,i)'; % 10 X 5000 and 26 X 5000
    
endfor


%input"Starting big_delta1");
for i=1:size(X,1)
  big_delta1 = big_delta1 + s_delta2(:,i)* X'(:,i)'; % 25 X 5000 and 401 X 5000
    
endfor

big_delta2 = big_delta2/m ;
%input("size of big_delta2");
%size(big_delta2);
%size(Theta2);
big_delta2 = [big_delta2(:,1) big_delta2(:,2:end) + (lambda/m).*Theta2(:,2:end)];
big_delta1 = big_delta1/m;
%input("size of big_delta1");
%size(big_delta1);
%size(Theta1);

big_delta1 = [big_delta1(:,1) big_delta1(:,2:end) + (lambda/m).*Theta1(:,2:end)];

grad = [big_delta1(:);big_delta2(:)];

%disp(grad);
%input("grad calculated");

%input"BackPropogation finished");





% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
