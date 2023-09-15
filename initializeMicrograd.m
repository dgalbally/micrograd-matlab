function initializeMicrograd
%==========================================================================
% initializeMicrograd.m
%==========================================================================
% Initialize the micrograd-matlab toolbox.
%
% This function initilizes the toolbox before its first use in a MATLAB
% session. The function adds all the necessary folders and subfolders to
% the MATLAB path to ensure that all classes and auxiliary functions are
% found.
%
% Usage: initializeMicrograd (i.e.  without arguments)
% Note: Run in the top micrograd-matlab folder.
% 
%==========================================================================
% The home path is set to the folder that contains this initialization file
homePath = fileparts(which(mfilename()));

% Set paths
addpath(homePath,...
    fullfile (homePath, 'src'),...
    fullfile (homePath, 'tests'));
addSubdirsToPath (fullfile (homePath, 'src'), filesep);
addSubdirsToPath (fullfile (homePath, 'tests'), filesep);

disp(' ');
disp('   ----------------------- micrograd-matlab --------------------');
disp('   See the README file for details.');
disp('   To get started, run some of the scripts in the "tests" folder')
disp('   -------------------------------------------------------------');
disp(' ');

end


function addSubdirsToPath(d,filesep)
    dl = dir(d);
    for i = 1:length(dl)
        if (dl(i).isdir)
            if      (~strcmp(dl(i).name,'.')) && ...
                    (~strcmp(dl(i).name,'..')) && ...
                    (~strcmp(dl(i).name(1),'@'))
                addpath(fullfile(d, dl(i).name))
                addSubdirsToPath(fullfile(d, dl(i).name), filesep);
            end
        end
    end
    return;
end