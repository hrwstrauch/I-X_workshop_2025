%% SINDy PDE find: the Kuramoto-Sivashinsky equation
% MATLAB version of 3_SINDy_PDE.ipynb (with the MATLAB port of PySINDy, package +ps).
%
% Kuramoto-Sivashinsky equation:   u_t = -u u_x - u_xx - u_xxxx
%
% It describes chaotic dynamics of laminar flame fronts (Sivashinsky 1977), reaction-diffusion systems
% (Kuramoto and Tsuzuki 1976) and coating flows (Sivashinsky and Michelson 1980).
%
% PDE-FIND was published by Rudy et al.:  https://www.science.org/doi/10.1126/sciadv.1602614
% Data and code from the PySINDy tutorial 10_PDEFIND_examples.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'utils'))
setup_pysindy

%% Load the data
% KS_data.mat holds the space grid x (1024), the times tt (251) and the field uu(x, t) (1024 x 251)
data = load(fullfile(here, '..', 'data', 'KS_data.mat'));
t = data.tt(:);
x = data.x(:);
u = data.uu;
dt = t(2) - t(1);
dx = x(2) - x(1);

% add Gaussian white noise to the data
rng(1)
sig = 0.0;                              % adjust the noise level
u = u + sig * std(u(:), 1) * randn(size(u));

% finite-difference time derivative: axis 2 is the time axis of u(x, t)
u_dot = ps.differentiation.FiniteDifference('axis', 2).call(u, dt);

% plot u and u_dot
figure('Position', [100 100 900 360])
subplot(1, 2, 1)
pcolor(t, x, u), shading('flat'), colorbar
xlabel('t'), ylabel('x'), title('u(x, t)')
subplot(1, 2, 2)
pcolor(t, x, u_dot), shading('flat'), colorbar
xlabel('t'), title('du/dt (x, t)')

%% Plot the spatial derivatives
% axis 1 is the space axis; d is the order of the derivative
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

%% Define the PDE library and identify the PDE
% A library that is quadratic in u and fourth order in the spatial derivatives of u.
% (The notebook builds it with PDELibrary(function_library=PolynomialLibrary(degree=2, include_bias=False),
% derivative_order=4, spatial_grid=x).  In the current PySINDy, and in this port, a PDELibrary holds the
% derivatives only, and the products with functions of u are written with the operators + and *:
%     poly        : u, u^2
%     pde         : u_x, u_xx, u_xxx, u_xxxx
%     poly * pde  : u u_x, u u_xx, ..., u^2 u_xxxx
% These are the 14 features of the notebook, in a different order.
poly    = ps.feature_library.PolynomialLibrary('degree', 2, 'include_bias', false);
pde     = ps.feature_library.PDELibrary('derivative_order', 4, 'spatial_grid', x);
pde_lib = poly + pde + poly * pde;

% sequentially thresholded ridge regression (alpha = 1e-5) with normalized columns
optimizer = ps.optimizers.STLSQ('threshold', 10, 'alpha', 1e-5, 'normalize_columns', true);
model = ps.SINDy('feature_library', pde_lib, 'optimizer', optimizer);
model.fit(u, dt, 'feature_names', {'u'});

% print the library
disp('PDE SINDy library:')
disp(strjoin(pde_lib.get_feature_names({'u'}), ',  '))

% print the identified PDE
disp('Identified PDE:')
model.print()
