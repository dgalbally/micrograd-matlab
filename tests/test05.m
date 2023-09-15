% Test to verify a binary classifier model built on top of a multilayer 
% preceptron neural network.
clear
close all
fprintf('\n')
fprintf('------------------------------------------------------------- \n')
fprintf('---------------- TEST BINARY CLASSIFIER MODEL --------------- \n')
fprintf('------------------------------------------------------------- \n')
fprintf('\n')

% Input matrix (spatial coordinates) of training data
X = [[ 1.82427454 -0.33587658]
     [ 0.93513636  0.18499181]
     [ 0.45033121 -0.27729713]
     [ 0.27971163 -0.05989899]
     [-0.86555078  0.02921278]
     [ 1.70363415 -0.27452339]
     [ 0.88676578 -0.55232135]
     [ 1.54220135 -0.2727132 ]
     [-0.88776955  0.6637943 ]
     [-1.14921596  0.15032604]
     [-0.38073845  0.82580922]
     [ 2.07678233  0.15870648]
     [ 0.11662765  0.23409138]
     [ 0.14418603  0.43872356]
     [-0.5363544   0.5995304 ]
     [ 0.85577726  0.5805372 ]
     [ 1.65451859 -0.28984739]
     [-0.58852157  0.73918606]
     [ 0.87234037 -0.31344718]
     [-0.30284441  0.79899662]
     [-0.58694304  0.67324567]
     [-0.97055562  0.30403289]
     [ 0.04586288  0.24073787]
     [ 0.55970769 -0.24396152]
     [ 1.35326918 -0.37416619]
     [-0.79805272  0.48731578]
     [ 1.80394414 -0.21563669]
     [ 0.33604268 -0.48267802]
     [ 0.2467885   1.1780853 ]
     [ 0.28455584  1.00615099]
     [ 1.21037563  0.22406922]
     [ 0.59555349 -0.42710161]
     [ 0.8291936  -0.56790715]
     [ 0.97631769  0.34746091]
     [ 0.66926793  0.7059185 ]
     [ 1.11142299  0.02127485]
     [ 2.10943472  0.61548847]
     [ 0.86776701  0.50459044]
     [ 0.01963056  1.10217204]
     [ 0.14964279 -0.05456159]];

% Output vector of training data
y = [ 1 -1  1  1 -1  1  1  1 -1 -1 -1  1  1  1 -1 -1  1 -1  1 -1 -1 -1  1  1, ...
      1 -1  1  1 -1 -1 -1  1  1 -1 -1 -1  1 -1 -1  1];

% Set seed of random number generator for reproducibility
rng('default')

% Expected results
yRef = 1.693397845574425;        % Expected output for (x,y) = (0,0)
success = true;

% Build binary classifier model
model = BinaryClassifier(X, y', [16, 16, 1]);

% Train model
model = Train(model);

% Report output for input point (0,0)
yModel = model.NeuralNet.Evaluate([0,0]);
yModel = yModel{1}.Data;
fprintf('\n Model output for x = (0,0): y = %f... \n', yModel)

% Calculate errors
e = abs(yRef - yModel);

% Generate plot
fprintf('\n Generating plot of training points and predicted regions... \n')
test05_Plot(X, y, model);

% Show errors
if e < 1E-7
    fprintf('\n Difference with expected value: %e. RESULT: OK. \n', e)
else
    fprintf('\n Maximum error: %e. RESULT: NOT OK. \n',e)
    success = false;
end

% Display error if any of the checks was not successful
if ~success
    error('Verification test 05 was not successful');
else
    fprintf('------------------------------------------------------------- \n')
    fprintf('VERIFICATION TEST 05 WAS COMPLETED SUCCESSFULLY. \n')
    fprintf('------------------------------------------------------------- \n')
end


function test05_Plot(X, y, model)
    nX = 25;
    nY = 25;
    
    xmin = min(X(:,1))-0.4;
    xmax = max(X(:,1))+0.4;
    ymin = min(X(:,2))-0.4;
    ymax = max(X(:,2))+0.4;
    xplot = linspace(xmin, xmax, nX);
    yplot = linspace(ymin, ymax, nY);
    [Xplot, Yplot] = meshgrid(xplot, yplot);
    
    Zplot = zeros(nY, nX);
    for i = 1:nY
        for j = 1:nX
            yModel = model.NeuralNet.Evaluate([Xplot(i,j), Yplot(i,j)]);
            if yModel{1}.Data > 0
                Zplot(i,j) = 1;
            else
                Zplot(i,j) = -1;
            end
        end
    end
    
    figure()
    hold on
    contourf(Xplot, Yplot, Zplot, 'edgecolor','none')
    colormap([0.3010 0.7450 0.9330; 0.9290 0.6940 0.1250])
    xlabel('X')
    ylabel('Y')
    
    plusIndices = find(y>0);
    minusIndices = find(y<0);

    scatter(X(plusIndices,1), X(plusIndices,2), 50, [0 0.4470 0.7410], 'filled')
    scatter(X(minusIndices,1), X(minusIndices,2), 50, [0.8500 0.3250 0.0980], 'filled')
end