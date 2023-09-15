% Similar to test 01, but more operations are tested.
clear
close all
fprintf('\n')
fprintf('------------------------------------------------------------- \n')
fprintf('------------- TEST FORWARD AND BACKPROPAGATION -------------- \n')
fprintf('--------------- OVER A DYNAMICALLY BUILT DAG ---------------- \n')
fprintf('------------------------------------------------------------- \n')
fprintf('\n')

% Expected results
gval = 2.5;
dgda = -27.5;
dgdb = -120;
success = true;

% Build DAG and calculate forward and backward passes
a = Value(-4.0);
b = Value(2.0);
c = a + b;
d = a * b + b^3;
c = c + 1;
c = 1 + c + (-a);
d = d * 2 + relu((b + a));
d = 3 * d + relu((b - a));
e = c - d;
f = e^2;
g = 10.0 / f;
g.Backward;

% Report results for forward pass
fprintf(' Result of forward pass: g = %f... \n', g.Data)

% Report two of the gradients calculated during the backward pass
fprintf(' Result of backward pass: Gradient dg/da = %f... \n', a.Grad)
fprintf(' Result of backward pass: Gradient dg/db = %f... \n', b.Grad)

% Calculate errors
e1 = abs(g.Data - gval);
e2 = abs(dgda - a.Grad);
e3 = abs(dgdb - b.Grad);
maxError = max([e1, e2, e3]);

% Show errors
if maxError < 1E-7
    fprintf('\n Difference with expected value: %e. RESULT: OK. \n', maxError)
else
    fprintf('\n Maximum error: %e. RESULT: NOT OK. \n',maxError)
    success = false;
end

% Display error if any of the checks was not successful
if ~success
    error('Verification test 02 was not successful');
else
    fprintf('------------------------------------------------------------- \n')
    fprintf('VERIFICATION TEST 02 WAS COMPLETED SUCCESSFULLY. \n')
    fprintf('------------------------------------------------------------- \n')
end
