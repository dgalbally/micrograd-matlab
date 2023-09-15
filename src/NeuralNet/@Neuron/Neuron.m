classdef  Neuron < Module
%==========================================================================
% Neuron.m
%==========================================================================
% This class is used to build individual neurons.
%
% This class is based on the 'Neuron' class described by Andrej Karpathy in 
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
        % ==================== Property: Weights ==========================
        % Cell array of 'Value' objects that contains the weights of the
        % neuron. The number of weights (length of the array of 'Weights') 
        % equals the number of inputs that the neuron can accept and is 
        % defined when the neuron is created inside the constructor. The
        % weights of the neuron are initialized by sampling from a uniform
        % distribution in interval [-1, 1].
        % =================================================================
        Weights

        % ====================== Property: Bias ===========================
        % Object of type 'Value' that defines the bias of the neuron. The
        % neurons built using this class are always initialized using zero
        % bias.
        % =================================================================
        Bias

        % ==================== Property: NonLinear ========================
        % Boolean that defines whether the output of the neuron is passed
        % through a nonlinear activation function (ReLU in our case) or
        % not.
        % =================================================================
        NonLinear
    end
    
% -------------------------------------------------------------------------
% CONSTRUCTOR METHOD
% -------------------------------------------------------------------------
    methods 
        
        function obj = Neuron(nInputs, varargin)
        % =================================================================
        % Function used to construct an object that belongs to the 'Neuron'
        % class.
        %
        % Inputs:
        % =======
        %    nInputs : Positive integer that defines the number of inputs 
        %              that the neuron accepts. This is the dimension of
        %              the neuron.
        %
        %    nonlinear : Boolean that specifies whether a nonlinear
        %                activation function (ReLU) is applied to the
        %                output of the neuron.
        %
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'Neuron' class.
        %
        % =================================================================
            % Define valid inputs
            validDimension = @(x) floor(x)==x && isscalar(x) && x>0;
            validNonLinear = @(x) islogical(x);
            default = true;
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'nInputs', validDimension);
            addOptional(p, 'nonlinear', default, validNonLinear);
            parse(p, nInputs, varargin{:});

            % Initialize weights using uniform random distribution
            weights = cell(1, p.Results.nInputs);
            for i = 1:length(weights)
                weights{i} = Value(-1 + 2*rand);
            end

            % Assign properties
            obj.Weights = weights;
            obj.Bias = Value(0);
            obj.NonLinear = p.Results.nonlinear;
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
        %   obj : Object that belongs to the 'Neuron' class.
        %
        % Outputs:
        % ========
        %    parameters : Cell array that contains the parameters of the 
        %                 neuron. The number of elements equals the number
        %                 of weights of the neuron plus 1. The last element 
        %                 of of the array is the bias. All other elements 
        %                 are the weights. All elements are 'Value' objects
        %
        % =================================================================
            parameters = obj.Weights;
            parameters{end+1} = obj.Bias;
        end

        function out = Evaluate(obj, x)
        % =================================================================
        % Function that returns the output of a neuron
        %
        % Inputs:
        % =======
        %   obj : Object that belongs to the 'Neuron' class.
        %
        %   x : Vector of doubles with the same length as the number of 
        %       weights that the neuron has. This input can also be a cell
        %       array of objects of type 'Value'.
        %
        % Outputs:
        % ========
        %    out : Object of type 'Value' that contains the value of the
        %          neuron evaluated for input 'x'. This is the dot product
        %          of the weights times x, plus the bias, and rectified in
        %          the case of nonlinear neurons.
        %
        % =================================================================
            % Convert to cell array if the input is a vector of doubles
            if ~iscell(x)
                x = num2cell(x);
            end

            % Calculate the output
            out = 0;
            for i = 1:length(obj.Weights)
                out = out + obj.Weights{i} * x{i};
            end
            out = out + obj.Bias;
            if obj.NonLinear
                out = relu(out);
            end
        end
        
    end
    
end

