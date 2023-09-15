% Test to verify the following functionalities:
%
%     * Creation of a dynamically built DAG (Directed Acyclic Graph) where
%       each downstream node is a value that is calculated by combining the 
%       values from other upstream nodes using mathematical operations.
%
%     * Forward propagation to calculate the value of any downstream node
%       based on the values of upstream nodes and the mathematical
%       operations defined between them.
%
%     * Backpropagation (reverse-mode autodiferentiation) to calculate the
%       gradient of any node with respect to the upstream nodes.

clear
close all
fprintf('\n')
fprintf('------------------------------------------------------------- \n')
fprintf('------------- TEST FORWARD AND BACKPROPAGATION -------------- \n')
fprintf('--------------- OVER A DYNAMICALLY BUILT DAG ---------------- \n')
fprintf('------------------------------------------------------------- \n')
fprintf('\n')

% Expected results
yval = -20.0;
dydx = 46.0;
success = true;

% Build DAG and calculate forward and backward passes
x = Value(-4.0);
z = 2 * x + 2 + x;
q = x.relu() + z * x;
h = relu(z*z);
y = h + q + q * x;
y.Backward;

% Report results for forward pass
fprintf(' Result of forward pass: y = %f... \n', y.Data)

% Report gradient dy/dx calculated during the backward pass
fprintf(' Result of backward pass: Gradient dy/dx = %f... \n', x.Grad)

% Calculate errors
e1 = abs(yval - y.Data);
e2 = abs(dydx - x.Grad);
maxError = max([e1, e2]);

% Show errors
if maxError < 1E-7
    fprintf('\n Difference with expected value: %e. RESULT: OK. \n', maxError)
else
    fprintf('\n Maximum error: %e. RESULT: NOT OK. \n',maxError)
    success = false;
end

% Display error if any of the checks was not successful
if ~success
    error('Verification test 01 was not successful');
else
    fprintf('------------------------------------------------------------- \n')
    fprintf('VERIFICATION TEST 01 WAS COMPLETED SUCCESSFULLY. \n')
    fprintf('------------------------------------------------------------- \n')
end
