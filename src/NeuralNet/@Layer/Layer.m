classdef  Layer < Module
%==========================================================================
% Layer.m
%==========================================================================
% This class is used to build individual a layer of neurons.
%
% This class is based on the 'Layer' class described by Andrej Karpathy in 
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
        % ==================== Property: Neurons ==========================
        % Cell array of 'Neuron' objects.
        % =================================================================
        Neurons
    end
    
% -------------------------------------------------------------------------
% CONSTRUCTOR METHOD
% -------------------------------------------------------------------------
    methods 
        
        function obj = Layer(nInputs, nOutputs, varargin)
        % =================================================================
        % Function used to construct an object that belongs to the 'Layer'
        % class.
        %
        % Inputs:
        % =======
        %    nInputs : Positive integer that defines the number of inputs 
        %              that the layer accepts. Inputs are transferred to
        %              all neurons that form the layer, so this also
        %              defines the number of inputs that each neuron will
        %              accept.
        %
        %    nOutputs : Positive integer that defines the number of outputs
        %               that the layer generates. This is equal to the
        %               number of neurons that form the layer.
        %
        %    varargin : These are the optional inputs that the constructor
        %               of the 'Neuron' class can accept (i.e., input 
        %               'nonlinear' that specifies whether the neuron is 
        %               linear or nonlinear).
        %
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'Layer' class.
        %
        % =================================================================
            obj.Neurons = cell(1, nOutputs);
            for i = 1:nOutputs
                obj.Neurons{i} = Neuron(nInputs, varargin{:});
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
        %   obj : Object that belongs to the 'Layer' class.
        %
        % Outputs:
        % ========
        %    parameters : Cell array that contains the parameters (weights
        %                 and biases) of all the neurons that form the 
        %                 layer. 
        %
        % =================================================================
            % Number of neurons
            nn = length(obj.Neurons);

            % Initialize parameters
            parameters = cell(1, nn);

            % Fill parameters
            for i = 1:nn
                parameters{i} = Parameters(obj.Neurons{i});
            end
            parameters = horzcat(parameters{:});
        end

        
        function out = Evaluate(obj, x)
        % =================================================================
        % Function that returns the output of the layer
        %
        % Inputs:
        % =======
        %   obj : Object that belongs to the 'Layer' class.
        %
        %   x : Vector of doubles with the same length as the number of 
        %       weights that each neuron has. This input can also be a cell
        %       array of objects of type 'Value'. Note that the same input
        %       'x' is passed to all the neurons that form the layer.
        %
        % Outputs:
        % ========
        %    out : Cell array of 'Value' objects, where the ith element of
        %          the cell array contains the output of the ith neuron of
        %          the layer.
        %
        % =================================================================
            % Convert to cell array if the input is a vector of doubles
            if ~iscell(x)
                x = num2cell(x);
            end

            % Calculate the output
            nn = length(obj.Neurons);
            out = cell(1, nn);
            for i = 1:nn
                out{i} = Evaluate(obj.Neurons{i}, x);
            end
        end
        
    end
    
end

