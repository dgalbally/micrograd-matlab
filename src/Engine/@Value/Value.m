classdef Value < handle
%==========================================================================
% Value.m
%==========================================================================
% This class is used to represent a single scalar variable at a node of a
% graph, and the gradient of any variable of the graph (usually the 
% variable at the final node of the graph or 'output') with respect to the
% variable. 
% 
% The graph is formed such that each variable only has a maximum of two 
% nodes upstream (two parents) and one node downstream (one child). For 
% example, the following graph, read from top to bottom would be used to 
% represent the combination of variables 'a' and 'b' to generate variable 
% 'c' and, subsequently, the combination of variable 'c' with variable 'd' 
% to generate variable 'e':
%
%     a   b
%      \ /
%       c   d
%        \ /
%         e
%
% The following sequence of operations could be represented with the graph
% shown above, by defining the appropriate operations between each node:
%
%   a = 2.0;
%   b = -3.0;
%   c = 4*a - b;
%   d = 10;
%   e = c*d;
%
% In the example above, all the nodes of the graph are objects that belong
% to the present 'Value' class. Note that the class inherits from 'handle', 
% so all objects that belong to the 'Value' class are passed by reference.
%
% This class is based on the 'Value' class described by Andrej Karpathy in 
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
        % ====================== Property: Data ===========================
        % Scalar numerical value that stores the value of the variable.
        % Type: double
        % =================================================================
        Data = 0;

        % ====================== Property: Grad ===========================
        % Scalar numerical value that stores the gradient of the output
        % node of the graph (or any other node specified by the user) 
        % relative to the variable defined in the current object.
        % Type: double
        % =================================================================
        Grad = 0;

        % ======================= Property: UUID ==========================
        % This property stores a string that uniquely identifies each 
        % object created by the 'Value' constructor. MATLAB does not give
        % direct access to memory addresses of handle objects, so this
        % identifier is needed for implementing an efficient graph 
        % topology creation algorithm in function 'buildTopo'. 
        % 
        % An alternative that does not require the use of this UUID would
        % be to define the 'visited' variable in 'buildTopo' as an array of
        % 'Value' objects call the method 'ismember' to determine if a 
        % given 'Value' object has been added to the 'visited' array. This 
        % alternative is very slow, so it has been discarded.
        % =================================================================
        UUID;
    end
    
    properties (SetAccess = private)
        
        % ==================== Property: m_backward =======================
        % This is a handle to a function that calculates the gradient of 
        % the output relative to the parent nodes of the object, based on 
        % the known gradient of the output with respect to the object (note
        % that the gradient of the output with respect to the object is
        % stored in property Value.Grad)
        % =================================================================
        m_backward
        
        % ====================== Property: m_prev =========================
        % This is a cell array that contains the parent variables of the
        % object (i.e., the variables that are combined to generate the 
        % object using a mathematical operation). This cell array can have 
        % a maximum of two elements (an object of type 'Value' can have a 
        % maximum of two parent variables and a minimum of zero). When the
        % graph is traversed backwards, this property contains the children
        % of the object.
        %
        % In the example shown at the beginning of this file, property
        % 'm_prev' for object 'c' would contain {a, b}. However, it would
        % be empty for object 'd'.
        % =================================================================
        m_prev
        
        % ======================== Property: m_op =========================
        % String that specifies the operation between the two parent
        % variables needed to generate the object. In the example graph
        % used at the beginning of this file, property 'm_op' of variable 
        % 'e' would contain the character '*', since parent variables 'c' 
        % and 'd' are multiplied to generate 'e'.
        %
        % This property is just used for debugging.
        % =================================================================
        m_op
        
    end
    
% -------------------------------------------------------------------------
% CONSTRUCTOR METHOD
% -------------------------------------------------------------------------       
    methods
        
        function obj = Value(data, previous, operation)
        % =================================================================
        % Function used to construct an object that belongs to the 'Value'
        % class.
        %
        % Inputs:
        % =======
        %    data: Scalar numerical value that stores the value of the 
        %          variable. This is an input of type double.
        %
        %    previous: Cell array that contains the objects that are 
        %              combined in order to generate the current object.
        %
        %    operation: String that specifies the mathematical operation 
        %               to be used for combining the objects stored in 
        %               'previous' in order to generate the object.
        %    
        % Outputs:
        % ========
        %    obj : Object that belongs to the 'Value' class.
        %
        % =================================================================
            % Default inputs
            if nargin == 1
                previous = {};
                operation = '';
            end

            % Avoid repeated parents (for example, if y = x*x, the value
            % 'x' should only appear once as a parent of 'y')
            if length(previous) == 2
                if strcmp(previous{1}.UUID, previous{2}.UUID)
                    previous = previous(1);
                end
            end

            % Assign properties
            obj.Data = data;
            obj.Grad = 0;
            obj.m_prev = previous;
            obj.m_op = operation;
            
            % Create UUID
            obj.UUID = matlab.lang.internal.uuid();
        end
    end
 
% -------------------------------------------------------------------------
% METHODS
% -------------------------------------------------------------------------
    methods
               
        function out = plus(a, b)
        % =================================================================
        % Method used to add two 'Value' objects or a 'Value' object and a
        % numerical scalar value. This is a method overload of the standard
        % 'plus' method implemented in MATLAB for performing additions of
        % doubles. Therefore, it can be called using the same syntax as the
        % original MATLAB method:
        %
        %    out = plus(Value, Value), or out = plus(Value, double) or 
        %    out = plus(double, Value)
        %
        %    out = Value + Value, or out = Value + double, or 
        %    out = double + Value
        %
        % Inputs:
        % =======
        %    a, b : Objects that belong to the 'Value' class or doubles.
        %    
        % Outputs:
        % ========
        %    out : Object that belongs to the 'Value' class. The 'Data'
        %          property is the sum of the values stored in 'a' and 'b'.
        %
        % =================================================================
            if isa(a, 'double')
                a = Value(a);
            end
            if isa(b, 'double')
                b = Value(b);
            end
            out = Value(a.Data + b.Data, {a, b}, '+');
            out.m_backward = @backward;

            function backward()
                a.Grad = a.Grad + out.Grad;
                b.Grad = b.Grad + out.Grad;
            end
        end


        function out = mtimes(a, b)
        % =================================================================
        % Method used to multiply two 'Value' objects or a 'Value' object 
        % and a numerical scalar value. This is a method overload of the 
        % standard 'mtimes' method implemented in MATLAB for multiplying
        % variables of type 'double'.
        %
        % Inputs:
        % =======
        %    a, b : Objects that belong to the 'Value' class or doubles.
        %    
        % Outputs:
        % ========
        %    out : Object that belongs to the 'Value' class. The 'Data'
        %          property is calculated by multiplying the values stored 
        %          in 'obj' and 'other'.
        %
        % =================================================================
            if isa(a, 'double')
                a = Value(a);
            end
            if isa(b, 'double')
                b = Value(b);
            end
            out = Value(a.Data * b.Data, {a, b}, '*');
            out.m_backward = @backward;

            function backward()
                a.Grad = a.Grad + b.Data * out.Grad;
                b.Grad = b.Grad + a.Data * out.Grad;
            end
        end


        function out = power(obj, b)
        % =================================================================
        % Method used to generate a Value object where the data property
        % equals obj.Data^other. This is a method overload of the standard
        % 'power' method implemented in MATLAB for raising a double to the
        % corresponding powers in another double (i.e., a^b).
        %
        % Inputs:
        % =======
        %    obj : Object that belongs to the 'Value' class.
        %
        %    b : Object that belongs to types 'double' or 'int'. This is
        %        exponent.
        %    
        % Outputs:
        % ========
        %    out : Object that belongs to the 'Value' class. The 'Data'
        %          property is calculated as obj.Data^b.
        %
        % =================================================================
            assert(isa(b, 'integer')||isa(b, 'double'), ...
                  'Only integer and double exponents are accepted for now')
            out = Value(obj.Data ^ b, {obj}, ['^',num2str(b)]);
            out.m_backward = @backward;

            function backward()
                obj.Grad = obj.Grad + (b * obj.Data^(b-1))*out.Grad;
            end
        end


        function out = relu(obj)
        % =================================================================
        % This method is an overload of the rectified linear unit (ReLU)
        % activation operation. This performs a nonlinear threshold
        % operation, where any input value less than zero is set to zero.
        %
        % Inputs:
        % =======
        %    obj : Object that belongs to the 'Value' class.
        %    
        % Outputs:
        % ========
        %    out : Object that belongs to the 'Value' class, calculated
        %          after applying the ReLU function to the input object.
        %
        % =================================================================
            if obj.Data < 0
                rect = 0;
            else
                rect = obj.Data;
            end
            out = Value(rect, {obj}, 'ReLU');
            out.m_backward = @backward;

            function backward()
                obj.Grad = obj.Grad + (out.Data > 0) * out.Grad;
            end
        end


        function obj = Backward(obj)
        % =================================================================
        % Method used to calculate the gradient of the object with respect
        % to all the previous nodes of the graph. The algorithm starts at
        % the object and travels the graph backwards. Therefore, the node
        % 'obj' is considered the parent node of the graph and all the
        % upstream nodes are considered children nodes. This method
        % calculates the property 'Grad' of each children node as the
        % gradient of the object, relative to the children node.
        %
        % In the example graph shown at the beginning of the file,
        % Backward(e) will travel the graph upwards and will calculate
        % the gradients de/dd, de/dc, de/db and de/da. These gradients will
        % be stored in the 'Grad' property of the object associated with
        % each node of the graph, d.Grad, c.Grad, b.Grad and a.Grad.
        %
        % Inputs:
        % =======
        %    obj : Object that belongs to the 'Value' class.
        %    
        % Outputs:
        % ========
        %    No outputs are generated. The method simply traverses all the
        %    children in the graph starting from 'obj', and updates the
        %    'Grad' property of those children.
        %
        % =================================================================

            % Define the topological order of all the children in the graph
            topo = {};
            visited = dictionary(string([]), true);
            buildTopo(obj)

            % Gradient of object relative to itself is always 1.0
            obj.Grad = 1;

            % Apply chain rule by traversing the graph backwards
            topo = flip(topo);
            for i = 1:length(topo)
                v = topo{i};
                v.m_backward();
            end

            function buildTopo(v)

                if ~isKey(visited, v.UUID)
                    visited(v.UUID) = true;
                    for j = 1:length(v.m_prev)
                        buildTopo(v.m_prev{j})
                    end
                    topo{end+1} = v;
                end
            end
                
        end  
        
    end


% -------------------------------------------------------------------------
% OPERATIONS DERIVED FROM PREVIOUS METHODS
% -------------------------------------------------------------------------
    methods
        
        function out = uminus(obj)
            out = obj * (-1);
        end

        function out = minus(a, b)
            out = a + (-b);
        end

        function out = mrdivide(a, b)
            out = a * b^-1;
        end

    end
    
end


