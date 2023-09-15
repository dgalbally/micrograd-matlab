classdef  BinaryClassifier
%==========================================================================
% BinaryClassifier.m
%==========================================================================
% This class is used to build a machine learning binary classifier model
% based on a multilayer perceptron neural network (MLP). This model is 
% adequate for classifying binary data, where the output can only take
% two values (binary data): -1 or 1.
% 
% For training, the objective function is defined using a SVM "Max-Margin" 
% binary classification loss, and Gradient Descent (SGD) is used for
% minimizing the loss.
% 
%==========================================================================
% -------------------------------------------------------------------------
% PROPERTIES
% -------------------------------------------------------------------------
    properties
        % =================== Property: NeuralNet =========================
        % This property contains a multilayer perceptron neural network
        % (object that belongs to class MLP).
        % =================================================================
        NeuralNet

        % ==================== Property: InputData ========================
        % This is a matrix of doubles with dimension [nPoints x dimIn]. 
        % Each row contains the coordinates of an input data point for
        % which the true output is known (outputs are stored in property
        % 'OutputData'. Therefore, the true inputs/outputs stored in 
        % properties 'InputData' and 'OutputData' can be used for training 
        % or for testing the accuracy of the model.
        % =================================================================
        InputData

        % =================== Property: OutputData ========================
        % This is a matrix of doubles with dimension [nPoints x dimOut]. 
        % The ith row of the matrix contains the output that corresponds to 
        % the input data point stored in obj.InputData(i,:). 
        % =================================================================
        OutputData
    end
    
% -------------------------------------------------------------------------
% CONSTRUCTOR METHOD
% -------------------------------------------------------------------------
    methods 
        
        function obj = BinaryClassifier(inputs, outputs, nOutputs)
        % =================================================================
        % Function used to construct an object that belongs to the 
        % 'BinaryClassifier' class.
        %
        % Inputs:
        % =======
        %    inputs : [nPoints x dimIn] matrix of doubles that contains a 
        %             set of inputs where the true outputs are known.
        %
        %    outputs : [nPoints x dimOut] matrix of doubles that contains a 
        %              set of outputs. The ith row of this matrix is the 
        %              output that corresponds to the input contained in
        %              the ith row of matrix 'inputs'.
        %
        %    nOutputs : Vector of integers where the ith element specifies
        %               the number of outputs that the ith layer of the 
        %               MLP will generate. This input is the same as the
        %               input 'nOutputs' required by the constructor of the
        %               MLP class.
        %
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'BinaryClassifier' class.
        %
        % =================================================================
            [nInPoints, dimIn] = size(inputs);
            [nOutPoints, dimOut] = size(outputs);
            assert(nInPoints == nOutPoints, ...
               'Number of input and output points must be the same')
            assert(dimOut == nOutputs(end), ...
               'Number of final layer neurons must equal output dimension')
            obj.NeuralNet = MLP(dimIn, nOutputs);
            obj.InputData = inputs;
            obj.OutputData = outputs;
        end
        
    end
 

% -------------------------------------------------------------------------
% METHODS
% -------------------------------------------------------------------------
    methods
        
        function [loss, accuracy] = Loss(obj, varargin)
        % =================================================================
        % Method that calculates the loss function of the model. This is
        % the difference between the outputs predicted by the model and the
        % true outputs stored in property obj.OutputData.
        %
        % Inputs:
        % =======
        %   obj : Object that belongs to the 'BinaryClassifier' class.
        %
        %   alpha : Optional parameter that will be used for L2
        %           regularization of the loss function. If not specified,
        %           a value of 1E-4 will be used.
        %
        %   indices : This are the indices of the input and output points 
        %             stored in properties 'InputData' and 'OutputData'
        %             respectively that will be used for calculating the
        %             loss. In some cases, it may be desirable to only use
        %             a subset of points for training and a different
        %             subset of points for validating the model. This
        %             requires defining the indices of the points that will
        %             be used for calculating the loss (error) depending on
        %             whether we're training the model or testing the
        %             accuracy of the model. If not specified, all points
        %             will be used everytime the loss function is
        %             calculated.
        %
        % Outputs:
        % ========
        %    loss : This is the total loss (data loss + regularized loss)
        %           using the SVM Maximum Margin loss algorithm.
        %
        %    accuracy : Percent accuracy of the model.
        %
        % =================================================================
            % Define valid inputs
            validObj = @(x) isa(obj, 'BinaryClassifier');
            validAlpha = @(x) isscalar(x) && x>0;
            validIndices = @(x) isvector(x);

            % Parse inputs
            p = inputParser;
            addRequired(p, 'obj', validObj);
            addOptional(p, 'alpha', 1E-4, validAlpha);
            addOptional(p, 'indices', [], validIndices);
            parse(p, obj, varargin{:});

            % Assign inputs
            alpha = p.Results.alpha;
            if isempty(p.Results.indices)
                indices = 1:1:size(obj.InputData, 1);
            else
                indices = p.Results.indices;
            end

            % Extract inputs and outputs
            nPoints = length(indices);
            x = obj.InputData(indices,:);
            yTrue = obj.OutputData(indices,:);

            % Compute model predictions
            yPred = cell(1, nPoints);
            for i = 1:nPoints
                yPred(i) = obj.NeuralNet.Evaluate(x(i,:));
            end

            % Compute error between true outputs and predicted using SVM
            % maximum margin loss function
            losses = cell(1, nPoints);
            for i = 1:nPoints
                losses{i} = relu(1 - yTrue(i) * yPred{i});
            end
            
            % dataLoss = sum(losses) * 1/length(losses);
            dataLoss = 0;
            for i = 1:length(losses)
                dataLoss = dataLoss + losses{i};
            end
            dataLoss = dataLoss * 1/length(losses);

            % Perform L2 regularization
            parameters = obj.NeuralNet.Parameters();
            nParameters = length(parameters);
            paramLosses = cell(1, nParameters);
            for i = 1:nParameters
                paramLosses{i} = parameters{i} * parameters{i};
            end

            % regLoss = alpha * sum(paramLosses);
            regLoss = 0;
            for i = 1:length(paramLosses)
                regLoss = regLoss + paramLosses{i};
            end
            regLoss = alpha * regLoss;

            % Compute total loss
            loss = dataLoss + regLoss;

            % Compute accuracy
            accuracy = zeros(1,nPoints);
            for i = 1:nPoints
                accuracy(i) = (yPred{i}.Data>0) == (yTrue(i)>0);
            end
            accuracy = sum(accuracy) / length(accuracy) * 100;
        end


        function obj = Train(obj, varargin)
        % =================================================================
        % Method that trains the model.
        %
        % Inputs:
        % =======
        %   obj : Object that belongs to the 'BinaryClassifier' class.
        %
        %   maxIter : Maximum number of training iterations to be
        %             performed. This is an optional argument. It not
        %             specified, 100 iterations will be performed.
        %
        %   accuracy : This is the target accuracy to be reached in % (for
        %              example: 95). If not specified, a target accuracy of
        %              100% will be used. Training will stop when target
        %              accuracy is reached, even if the maximum number of
        %              iterations specified in 'maxIter' has not been
        %              reached.
        %
        %   alpha : Optional parameter that will be used for L2
        %           regularization of the loss function. If not specified,
        %           a value of 1E-4 will be used.
        %
        %   nPoints : Positive integer that specifies the number of points
        %             that will be used for training. If not specified, all
        %             the points stored in InputData/OutputData will be
        %             used.
        %
        % Outputs:
        % ========
        %    obj : Trained model (the neuron parameters have been adjusted
        %          to replicate the behavior of the training data).
        %
        % =================================================================
            % Define valid inputs
            validObj = @(x) isa(obj, 'BinaryClassifier');
            validMaxIter = @(x) floor(x)==x && isscalar(x) && x>0;
            validAccuracy = @(x) isscalar(x) && x>0;
            validAlpha = @(x) isscalar(x) && x>0;
            validNpoints = @(x) floor(x)==x && isscalar(x) && x>0;

            % Parse inputs
            p = inputParser;
            addRequired(p, 'obj', validObj);
            addOptional(p, 'maxIter', 100, validMaxIter);
            addOptional(p, 'accuracy', 100, validAccuracy);
            addOptional(p, 'alpha', 1E-4, validAlpha);
            addOptional(p, 'nPoints', [], validNpoints);
            parse(p, obj, varargin{:});

            % Assign inputs
            maxIter = p.Results.maxIter;
            targetAccuracy = p.Results.accuracy;
            alpha = p.Results.alpha;
            nPoints = p.Results.nPoints;

            % Extract indices of points that will be used for training
            indices = 1:1:size(obj.InputData, 1);
            if ~isempty(nPoints)
                indices = randperm(length(indices));
                indices = indices(1:nPoints);
            end

            % Training loop
            for k = 1:maxIter

                % Forward pass
                fprintf('Performing forward pass for iteration: %d \n', k)
                [loss, accuracy] = Loss(obj, alpha, indices);

                % Check if target accuracy has been reached
                if accuracy >= targetAccuracy
                    fprintf('Exit criteria reached: accuracy = %.1f, at iteration %d \n', accuracy, k)
                    break
                end

                % Set gradient of all neural network parameters to zero
                obj.NeuralNet = obj.NeuralNet.ZeroGradient;

                % Backward pass
                fprintf('Performing backward pass for iteration: %d \n', k)
                loss.Backward;

                % Update model parameters using SGD
                learningRate = 1.0 - 0.9*k/100;
                p = obj.NeuralNet.Parameters();
                for i = 1:length(p)
                    p{i}.Data = p{i}.Data - learningRate * p{i}.Grad;
                end

                % Print status
                fprintf('Iteration %d completed. Loss: %.8f, Accuracy: %.1f \n', k, loss.Data, accuracy)
            end
        end
        
    end
    
end