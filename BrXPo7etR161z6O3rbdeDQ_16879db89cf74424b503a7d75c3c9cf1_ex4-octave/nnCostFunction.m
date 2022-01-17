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

% convert y values to column vectors
yVec = zeros(m, num_labels);
for i = 1:m;
    yVec(i, y(i)) = 1;
endfor

% cost function
%add bias ones to X
X = [ones(m,1) X]; % 5000 x 401

z_2 = X * Theta1';
a_2 = sigmoid(z_2);   

a_2 = [ones(size(a_2,1),1) a_2]; %add bias ones
z_3 = a_2 * Theta2'; 
h = sigmoid(z_3); % output activations [0 0 0 0.99 0 0 0 0.01 0 0]
newCosts = sum((-yVec .* log(h)) - ((1 - yVec) .* log(1 - h)));
% size(Theta1) %25 x 401
% size(Theta2) % 10 x 26
% size(yVec) % 5000 x 10
% size(h) % 5000 x 10
% pause
J = sum(newCosts);

%return values
J = J / m;                  % cost function

%Regularisation code for J
temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0;
temp2(:,1) = 0;
regSum = sum(sum(temp1 .^ 2)) + sum(sum(temp2 .^ 2));
regCost = (lambda / (2 * m)) * regSum;
J = J + regCost;


% backpropagation
Theta2 = Theta2(:,2:end); %remove bias values

delta3 = h - yVec;          
delta2 = (delta3 * Theta2) .* sigmoidGradient(z_2); 

Theta1_grad = (delta2' * X) ./ m; 
Theta2_grad = (delta3' * a_2) ./ m;
% size(Theta1_grad) % 25 x 401
% size(Theta2_grad) % 10 x 26
% pause



% delta2                  %no
% delta3                  %no
% Theta1_grad             %no
% Theta2_grad             %no
% z_2                     %yes
% sigmoidGradient(z_2)    %yes
% a_2                     %yes
% h                         %yes
% yVec

%regularisation of the backpropagation deltas
Theta1(:,1) = 0;
Theta2 = [zeros(size(Theta2,1),1) Theta2];
Theta1_grad = Theta1_grad + Theta1 .* (lambda/m);
Theta2_grad = Theta2_grad + Theta2 .* (lambda/m);

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
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
