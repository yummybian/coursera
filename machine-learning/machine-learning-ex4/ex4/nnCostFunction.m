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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

A1 = [ones(m, 1) X];
 
del1_sum = 0;
del2_sum = 0;

for i=1:m
    % feedforward
    a1 = A1(i, :)
    z2 = Theta1 * a1'
    a2 = sigmoid(z2);
    a2 = vertcat(ones(1, 1), a2);
    a3 = sigmoid(Theta2 * a2);
    this_y = zeros(num_labels, 1);
    this_y(y(i)) = 1;
    j = sum((-this_y) .* log(a3) - (1 - this_y) .* log(1 - a3));
    J = J + j;
    
    % backforward
    delta3 = a3 - this_y
    delta2 = Theta2' * delta3 .* sigmoidGradient([1; z2]);
    delta2 = delta2(2:end);
    del1_sum = del1_sum + delta2 * a1;
    del2_sum = del2_sum + delta3 * a2';
    
end

J = J / m
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta1_pow_2 = Theta1 .* Theta1;
Theta2_pow_2 = Theta2 .* Theta2;
Reg = (lambda / (2*m)) * (sum(Theta1_pow_2(:)) + sum(Theta2_pow_2(:)));
J = J + Reg;


Theta1_grad = del1_sum / m;
Theta2_grad = del2_sum / m;


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
