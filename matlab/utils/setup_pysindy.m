function setup_pysindy()
%SETUP_PYSINDY  Put the MATLAB port of pysindy (package +ps) on the path.
%
%   The scripts of this folder call it first.  It looks for the folder pysindy_matlab in
%     1. the environment variable PYSINDY_MATLAB,
%     2. ~/data/GitHub/pysindy_matlab,
%     3. a folder pysindy_matlab next to the folder of this workshop.
%   Edit the list below if the port lives somewhere else.
if ~isempty(which('ps.SINDy')), return, end
homedir = getenv('HOME');
here = fileparts(mfilename('fullpath'));
cands = {getenv('PYSINDY_MATLAB'), fullfile(homedir, 'data', 'GitHub', 'pysindy_matlab'), ...
         fullfile(here, '..', '..', '..', 'pysindy_matlab'), fullfile(here, '..', '..', 'pysindy_matlab')};
for k = 1:numel(cands)
    if ~isempty(cands{k}) && isfolder(fullfile(cands{k}, '+ps'))
        addpath(cands{k});
        return
    end
end
error('setup_pysindy:notFound', ['The MATLAB port of pysindy (pysindy_matlab) was not found.\n' ...
    'Clone it, then add its folder to the path (addpath) or set the environment variable PYSINDY_MATLAB.']);
end
