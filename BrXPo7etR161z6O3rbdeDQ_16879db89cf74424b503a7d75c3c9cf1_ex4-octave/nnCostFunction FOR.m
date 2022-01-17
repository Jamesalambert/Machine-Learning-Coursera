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

%accumulator for Deltas
Acc1 = zeros(hidden_layer_size, input_layer_size + 1);
Acc2 = zeros(num_labels,hidden_layer_size + 1);

for i = 1:m;
    
% cost function
    %add zeroes to X
    Xi = [1 X(i,:)]; % 1 x 401

    z_2 = Xi * Theta1'; % 1 x 25
    a_2 = sigmoid(z_2); % 1 x 25   

    a_2 = [1 a_2]; % 1 x 26 add bias ones
    z_3 = a_2 * Theta2'; % 1 x 10
    h = sigmoid(z_3); % 1 x 10 output activations [0 0 0 0.99 0 0 0 0.01 0 0]
    newCosts = (-yVec(i,:) .* log(h)) - ((1 - yVec(i,:)) .* log(1 - h));
    % yVec(i,:)
%     h
%     newCosts
%     pause
    J = J + sum(newCosts);



% backpropagation
    temp_Theta2 = Theta2(:,2:end); %10 x 25
    temp_a_2 = a_2;%(:,2:end);
    temp_z_2 = z_2;
    
    delta3 = h - y(i);          % 1 x 10
    delta2 = (delta3 * temp_Theta2) .* sigmoidGradient(temp_z_2); % 25 x 1
    % size(temp_z_2)
%     size(delta2)
%     size(delta3)
%     size(temp_a_2)
%     size(X(i,:))
%     pause
    Acc2 = Acc2 + delta3' * temp_a_2;
    Acc1 = Acc1 + delta2' * Xi;
    
endfor

%return values
J = J / m;                  % cost function
  
Theta1_grad = Acc1 / m;     % grads
Theta2_grad = Acc2 / m;
% size(Theta1_grad)
% size(Theta2_grad)
% pause

%Regularisation code for J
temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0;
temp2(:,1) = 0;
regSum = sum(sum(temp1 .^ 2)) + sum(sum(temp2 .^ 2));
regCost = (lambda / (2 * m)) * regSum;
J = J + regCost;

%tests
% if ~size_equal(J, 1);
%     print('J is the wrong size')
%     pause
% endif
%
% if ~size_equal(h, zeros(1,10));
%     print('h is the wrong size')
%     pause
% endif

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
