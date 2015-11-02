function step = update_passive_aggressive(X_i, y_i, w)
% Computes the update step using a passive aggressive approach
%
% Usage:
%     step = update_passive_aggressive(X_i, y_i, w)
%
% Takes a 1 x (D+1) matrix X_i representing the current example,
% a scalar +1/-1 y_i correct label for that example
% and the current (D+1)x1 weights of the perceptron as arguments
%
% Returns the magnitude of the step in the direction of X_i that should be
% taken by the perceptron


%% YOUR CODE GOES HERE
%determine L first
r = y_i*dot(w,X_i);

if r >= 1
    L = 0;
else
    L = 1 - r;
end

eta = L/sum(X_i.^2);

step = eta*y_i;