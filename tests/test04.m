% Test to verify the construction and evaluation of neurons.

clear
close all
fprintf('\n')
fprintf('------------------------------------------------------------- \n')
fprintf('---------- TEST NEURON CONSTRUCTION AND EVALUATION ---------- \n')
fprintf('------------------------------------------------------------- \n')
fprintf('\n')

% Input vector
x = [1.0, -2.0];

% Set seed of random number generator for reproducibility
rng('default');

% Expected results
y = -0.993720375516119;
success = true;

% Build 2D neuron
neuron = Neuron(2, 'nonlinear', false);

% Evaluate neuron
out = Evaluate(neuron, x);

% Report output
fprintf(' Neuron output: y = %f... \n', out.Data)

% Calculate errors
e = abs(y - out.Data);

% Show errors
if e < 1E-7
    fprintf('\n Difference with expected value: %e. RESULT: OK. \n', e)
else
    fprintf('\n Maximum error: %e. RESULT: NOT OK. \n',e)
    success = false;
end

% Display error if any of the checks was not successful
if ~success
    error('Verification test 04 was not successful');
else
    fprintf('------------------------------------------------------------- \n')
    fprintf('VERIFICATION TEST 04 WAS COMPLETED SUCCESSFULLY. \n')
    fprintf('------------------------------------------------------------- \n')
end
