%% SINDy weak form PDE find
% MATLAB version of 6_weak_SINDy_PDE.ipynb (with the MATLAB port of PySINDy, package +ps).
%
% Kuramoto-Sivashinsky equation:   u_t = -u u_x - u_xx - u_xxxx
%
% With noisy data the derivatives of the data (here u_t and four space derivatives) are inaccurate.  The weak form
% avoids them: the equation is multiplied by a smooth test function that vanishes at the border of a small
% space-time box (a "subdomain"), and integrated over the box; the derivatives are moved onto the test function
% by partial integration.  Many random subdomains give the equations for the regression.
%
% Data and code from the PySINDy tutorial 12_weakform_SINDy_examples.
% The random numbers are MATLAB's, not numpy's: the noise and the subdomains differ from the notebook.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'utils'))
setup_pysindy

%% A library with custom functions (as in the notebook; the model is not used below)
poly_library = ps.feature_library.PolynomialLibrary();            % degree 2, with the constant term
custom_functions = {@(x) abs(x + 1.0), @(x) abs(x - 1)};
custom_lib = ps.feature_library.CustomLibrary(custom_functions) + poly_library;
model = ps.SINDy('feature_library', custom_lib); %#ok<NASGU>

%% Load the data
data = load(fullfile(here, '..', 'data', 'KS_data.mat'));
t = data.tt(:);
x = data.x(:);
u = data.uu;                            % u(x, t): 1024 x 251
dt = t(2) - t(1);
dx = x(2) - x(1);

% add Gaussian white noise to the data
rng(1)
sig = 0.01;                             % the noise level (the notebook sets 0.0 first and then 0.01)
u = u + sig * std(u(:), 1) * randn(size(u));

% finite-difference time derivative (axis 2 is time): with noise it is useless
u_dot = ps.differentiation.FiniteDifference('axis', 2).call(u, dt);

figure('Position', [100 100 900 360])
subplot(1, 2, 1)
pcolor(t, x, u), shading('flat'), colorbar
xlabel('t'), ylabel('x'), title('u(x, t)')
subplot(1, 2, 2)
pcolor(t, x, u_dot), shading('flat'), colorbar
xlabel('t'), title('du/dt (x, t)')

%% Plot the spatial derivatives of the noisy data
ux    = ps.differentiation.FiniteDifference('d', 1, 'axis', 1).call(u, dx);
uxx   = ps.differentiation.FiniteDifference('d', 2, 'axis', 1).call(u, dx);
uxxx  = ps.differentiation.FiniteDifference('d', 3, 'axis', 1).call(u, dx);
uxxxx = ps.differentiation.FiniteDifference('d', 4, 'axis', 1).call(u, dx);

figure('Position', [100 100 1400 340])
derivs = {ux, uxx, uxxx, uxxxx};
names = {'u_x(x, t)', 'u_{xx}(x, t)', 'u_{xxx}(x, t)', 'u_{xxxx}(x, t)'};
for k = 1:4
    subplot(1, 4, k)
    pcolor(t, x, derivs{k}), shading('flat')
    xlabel('t')
    if k == 1, ylabel('x'); else, set(gca, 'YTickLabel', []); end
    title(names{k})
end

%% The PDE library, and the identification with finite differences
% A library that is quadratic in u and fourth order in the spatial derivatives of u:
%     poly        : u, u^2
%     pde         : u_x, u_xx, u_xxx, u_xxxx
%     poly * pde  : u u_x, u u_xx, ..., u^2 u_xxxx
% (The notebook writes this as PDELibrary(function_library=PolynomialLibrary(degree=2, include_bias=False),
% derivative_order=4, spatial_grid=x); here a PDELibrary holds the derivatives only, see SINDy_3_PDE.m.)
poly    = ps.feature_library.PolynomialLibrary('degree', 2, 'include_bias', false);
pde     = ps.feature_library.PDELibrary('derivative_order', 4, 'spatial_grid', x);
pde_lib = poly + pde + poly * pde;

% sequentially thresholded ridge regression (alpha = 1e-5) with normalized columns
optimizer = ps.optimizers.STLSQ('threshold', 10, 'alpha', 1e-5, 'normalize_columns', true);
model = ps.SINDy('feature_library', pde_lib, 'optimizer', optimizer);
model.fit(u, dt, 'feature_names', {'u'});

% print the identified PDE
disp('Identified PDE:')
model.print()

%% The weak form
% The space-time grid is needed: XT(i, j, :) = [x_i, t_j]
[X, T] = ndgrid(x, t);
XT = cat(3, X, T);

% the weak-form model: the same library, K = 1000 random subdomains (the default is 100); every subdomain is
% 1/20 of the extent of the data in x and in t (the half-widths H_xt can be given in fit)
rng(1)
weak_model = ps.WeakSINDy('feature_library', pde_lib, 'n_subdomains', 1000);
weak_model.fit(u, XT, 'feature_names', {'u'});

% print the identified PDE
disp('Identified PDE using weak form SINDy:')
weak_model.print()
