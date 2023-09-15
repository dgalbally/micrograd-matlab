classdef  Module
%==========================================================================
% Module.m
%==========================================================================
% This class is the base class that is used for building neurons, layers
% and multilayer perceptron (MLP) neural networks.
%
% This class is based on the 'Module' class described by Andrej Karpathy in 
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
        
    end
    
% -------------------------------------------------------------------------
% CONSTRUCTOR METHOD
% -------------------------------------------------------------------------
    methods 
        
        function obj = Module()
        % =================================================================
        % Function used to construct an object that belongs to the 'Module'
        % class.
        %
        % Inputs:
        % =======
        %    
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'Module' class.
        %
        % =================================================================
        end
        
    end
       
% -------------------------------------------------------------------------
% METHODS
% -------------------------------------------------------------------------
    methods
        
        function obj = ZeroGradient(obj)
        % =================================================================
        % Sets the gradient of all the parameters of the object equal to
        % zero.
        %
        % Inputs:
        % =======
        %    obj : Object that belongs to the 'Module' class.
        %
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'Module' class, where all the
        %          parameters have been set to zero.
        %
        % =================================================================
            parameters = obj.Parameters;
            for i = 1:length(parameters)
                parameters{i}.Grad = 0;
            end
        end
        
    end

% -------------------------------------------------------------------------
% STATIC METHODS
% -------------------------------------------------------------------------
    methods (Static = true)
        
        function parameters = Parameters()
        % =================================================================
        % Function that returns the parameters of the object.
        %
        % Inputs:
        % =======
        %
        % Outputs:
        % ========
        %    parameters : Cell array that contains all the parameters of
        %                 the object.
        %
        % =================================================================
            parameters = [];
        end
        
    end
    
end

