classdef  MLP < Module
%==========================================================================
% MLP.m
%==========================================================================
% This class is used to build a multilayer perceptron neural network (MLP)
% formed by at least three fully connected layers of neurons: an input and 
% an output layer with one or more hidden layers between the input and
% ouput layers. All the layers are formed by nonlinear neurons except the
% last layer, which is formed by linear neurons.
%
% This class is based on the 'MLP' class described by Andrej Karpathy in 
% his YouTube video 'The spelled-out intro to neural networks and 
% backpropagation: building micrograd', as well as the associated GitHub
% repository. See the following links:
%    * Video: https://www.youtube.com/watch?v=VMj-3S1tku0
%    * Repo: https://github.com/karpathy/micrograd
% 
%==========================================================================
% -------------------------------------------------------------------------
% PROPERTIES
% -------------------------------------------------------------------------
    properties
        % ==================== Property: Nvalues ==========================
        % Vector of integers that specifies the number of different data
        % values that flow between each layer of the perceptron. The length
        % of this vector equals the number of layers of the perceptron plus
        % one.
        %
        % For example, if we have that obj.Nvalues = [2, 16, 16, 1]. This
        % means that the perceptron has 3 layers:
        %
        %   * The first layer receives 2 inputs and generates 16 outputs. 
        %   * The second layer receives 16 inputs and generates 16 outputs.
        %   * The third layer receives 16 inputs and generates 1 output.
        %
        % Note that the number of inputs that the layer accepts equals that
        % number of inputs that each neuron of the layer accepts, since all
        % the neurons receive the same inputs. Also, the number of outputs
        % that the layer generates equals the number of neurons contained
        % in the layer, since each neuron only generates one output.
        % =================================================================
        Nvalues

        % ===================== Property: Layers ==========================
        % Cell array of 'MLP' objects.
        % =================================================================
        Layers
    end
    
% -------------------------------------------------------------------------
% CONSTRUCTOR METHOD
% -------------------------------------------------------------------------
    methods 
        
        function obj = MLP(nInputs, nOutputs)
        % =================================================================
        % Function used to construct an object that belongs to the 'MLP'
        % class.
        %
        % Inputs:
        % =======
        %    nInputs : Positive integer that defines the number of inputs 
        %              that the first layer accepts.
        %
        %    nOutputs : Vector of integers where the ith element specifies
        %               the number of outputs that the ith layer generates
        %               (this is equal to the number of neurons that form
        %               the layer).
        %
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'MLP' class.
        %
        % =================================================================
            obj.Nvalues = [nInputs, nOutputs];
            nLayers = length(nOutputs);
            obj.Layers = cell(1, nLayers);
            nonlinear = true;
            for i = 1:nLayers
                if i == nLayers
                    nonlinear = false;
                end
                obj.Layers{i} = Layer(obj.Nvalues(i), obj.Nvalues(i+1),...
                                      nonlinear);
            end

        end
        
    end
 

% -------------------------------------------------------------------------
% METHODS
% -------------------------------------------------------------------------
    methods
        
        function parameters = Parameters(obj)
        % =================================================================
        % Function that returns the parameters of the object.
        %
        % Inputs:
        % =======
        %   obj : Object that belongs to the 'MLP' class.
        %
        % Outputs:
        % ========
        %    parameters : Cell array that contains the parameters (weights
        %                 and biases) of all the neurons that form the 
        %                 MLP. 
        %
        % =================================================================
            % Number of layers
            nLayers = length(obj.Layers);

            % Initialize parameters
            parameters = cell(1, nLayers);

            % Calculate parameters of each layer
            for i = 1:nLayers
                parameters{i} = Parameters(obj.Layers{i});
            end
            parameters = horzcat(parameters{:});
        end

        
        function out = Evaluate(obj, x)
        % =================================================================
        % Function that returns the output of the MLP.
        %
        % Inputs:
        % =======
        %   obj : Object that belongs to the 'MLP' class.
        %
        %   x : Vector of doubles with the same length as the number of 
        %       inputs accepted by the first layer of neurons. This input 
        %       can also be a cell array of objects of type 'Value'. 
        %
        % Outputs:
        % ========
        %    out : Cell array of 'Value' objects, where the ith element of
        %          the cell array contains the output of the ith neuron of
        %          the final layer.
        %
        % =================================================================
            % Convert to cell array if the input is a vector of doubles
            if ~iscell(x)
                x = num2cell(x);
            end

            % Calculate the output
            nLayers = length(obj.Layers);
            for i = 1:nLayers
                x = Evaluate(obj.Layers{i}, x);
            end
            out = x;
        end
        
    end
    
end

