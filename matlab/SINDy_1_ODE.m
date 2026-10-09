%% SINDy model identification on Lorenz system data
% MATLAB version of 1_SINDy_ODE.ipynb.  This tutorial shows how to use SINDy (Sparse Identification of
% Nonlinear Dynamics) to discover the governing equations of the Lorenz system from noisy data.
%
%   xdot = sigma (y - x)
%   ydot = x (rho - z) - y
%   zdot = x y - beta z
%
% Steps:
%   1. generate noisy data: integrate the Lorenz ODE and add Gaussian white noise
%   2. compute derivatives by finite differences
%   3. build the library Theta, here polynomials up to order 3
%   4. compute the sparse regression
%   5. evaluate the identified SINDy model for forecasting
%   6. run the same with the MATLAB port of PySINDy (package +ps, see setup_pysindy.m)
%
% The random numbers are MATLAB's, not numpy's, so noisy results differ in detail from the notebook.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'utils'))        % lorenz_system, plot_sindy, setup_pysindy
setup_pysindy                           % finds the MATLAB port of pysindy

%% 1. Generate noisy Lorenz system data
rng(100)

% ode45 is the Runge-Kutta 4(5) method; tight tolerances as in the notebook
options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);

% Lorenz system parameters
sigma = 10;  rho = 28;  beta = 8/3;
params = [sigma, rho, beta];

% initial condition x0, time step dt and the time grid
x0 = [-8; 8; 27];
dt = 0.01;
t_eval = (0:dt:10).';

% solve the initial value problem
[~, x_true] = ode45(@(t, x) lorenz_system(t, x, params), t_eval, x0, options);

% add Gaussian white noise (np.std of the whole array is the population standard deviation: std(., 1))
rng(1)
sig = 0.0;
x_noisy = x_true + sig * std(x_true(:), 1) * randn(size(x_true));

% plot the noisy time series data
figure('Position', [100 100 560 300])
labels = {'x', 'y', 'z'};  colors = {'b', 'r', 'g'};
for i = 1:3
    subplot(3, 1, i)
    plot(t_eval, x_noisy(:, i), colors{i}), ylabel(labels{i})
    if i < 3, set(gca, 'XTickLabel', []); else, xlabel('time'); end
    if i == 1, title('Lorenz System Data'), end
end

%% 2. Compute derivatives (finite difference)
% fourth-order central difference
dx_dt = (1 / (12 * dt)) * (-x_noisy(6:end, :) + 8 * x_noisy(5:end-1, :) - 8 * x_noisy(3:end-3, :) + x_noisy(2:end-4, :));
x = x_noisy(4:end-2, :);                % cut the tails
t = t_eval(4:end-2);

%% 3. Build the library of nonlinear terms
% polynomials up to order 3 (poolData is defined at the end of this file)
polyorder = 3;
theta = poolData(x, polyorder);

fprintf('the library matrix theta has %d rows (time steps), and %d columns (nonlinear library terms).\n', ...
    size(theta, 1), size(theta, 2));
disp(theta(1:3, 1:6))                   % the first rows and columns

%% 4. Compute the sparse regression
% sequential thresholded least squares (sparsifyDynamics is defined at the end of this file)
threshold = 0.1;                        % the sparsification parameter: coefficients below it are set to zero
n_vars = size(x_true, 2);               % number of states: 3
xi = sparsifyDynamics(theta, dx_dt, threshold, n_vars);

% print the identified model: one row per library term, one column per equation
disp('Identified SINDy model coefficients (columns: xdot, ydot, zdot):')
disp(round(xi, 4))

%% 5. Evaluate the performance of the SINDy model for forecasting
sindy_params.xi = xi;
sindy_params.polyorder = polyorder;

% test trajectory from a new initial condition
x0_test = [8; 7; 15];
t_test = t_eval;

[~, x_test_true] = ode45(@(t, x) lorenz_system(t, x, params), t_test, x0_test, options);

% simulate the learned model
[t_sim, x_sindy_sim] = ode45(@(t, x) sindy_ode(t, x, sindy_params), t_eval, x0_test, options);

% compare the test data and the SINDy prediction
plot_sindy(t_test, x_test_true, t_sim, x_sindy_sim)

%% 6. PySINDy (the MATLAB port: package +ps)
% the default model: STLSQ (threshold = 0.1), a polynomial library of degree 2 and
% centred second-order finite differences for x_dot
model = ps.SINDy();
model.fit(x, dt, 'feature_names', {'x', 'y', 'z'});
model.print()

%%
% evolve the new initial condition in time with the SINDy model
x_test_pysindy = model.simulate(x0_test, t_test);

% compare the original data and the PySINDy prediction
plot_sindy(t_test, x_test_true, t_test, x_test_pysindy)

%% Reproduce the results above: change the order of the finite difference
% optimizer: sequentially thresholded least squares
optimizer = ps.optimizers.STLSQ('threshold', 0.2);
% differentiation method: centred difference, 4th order (or give the derivatives directly:
% model.fit(x, dt, 'x_dot', dx_dt))
differentiation_method = ps.differentiation.FiniteDifference('order', 4);
% library: polynomials of degree 3 (the default library has degree 2)
feature_library = ps.feature_library.PolynomialLibrary('degree', 3);

model = ps.SINDy('optimizer', optimizer, ...
                 'differentiation_method', differentiation_method, ...
                 'feature_library', feature_library);
model.fit(x, dt, 'feature_names', {'x', 'y', 'z'});
model.print()

%%
x_test_pysindy = model.simulate(x0_test, t_test);
plot_sindy(t_test, x_test_true, t_test, x_test_pysindy)

%% Local functions
function theta = poolData(x, polyorder)
% Builds a polynomial library of candidate functions: 1, x_i, x_i x_j, x_i x_j x_k (i <= j <= k).
[n_samples, n_vars] = size(x);
theta = ones(n_samples, 1);                              % the constant term
for i = 1:n_vars                                         % order 1
    theta = [theta, x(:, i)]; %#ok<AGROW>
end
if polyorder >= 2                                        % order 2
    for i = 1:n_vars
        for j = i:n_vars
            theta = [theta, x(:, i) .* x(:, j)]; %#ok<AGROW>
        end
    end
end
if polyorder >= 3                                        % order 3
    for i = 1:n_vars
        for j = i:n_vars
            for k = j:n_vars
                theta = [theta, x(:, i) .* x(:, j) .* x(:, k)]; %#ok<AGROW>
            end
        end
    end
end
end

function xi = sparsifyDynamics(theta, dx_dt, threshold, n_vars)
% Sequential thresholded least squares (STLS).
xi = theta \ dx_dt;                                      % initial guess: least squares
for k = 1:10                                             % iterative thresholding
    small_coeffs = abs(xi) < threshold;                  % find small coefficients
    xi(small_coeffs) = 0;                                % and threshold them
    for i = 1:n_vars                                     % n_vars is the state dimension
        big = ~small_coeffs(:, i);
        % regress the dynamics onto the remaining terms
        xi(big, i) = theta(:, big) \ dx_dt(:, i);
    end
end
end

function dx = sindy_ode(~, x, p)
% Right-hand side of the identified model: dx = Theta(x) * Xi.
theta_ode = poolData(x.', p.polyorder);                  % x is a column, poolData wants rows
dx = (theta_ode * p.xi).';
end
