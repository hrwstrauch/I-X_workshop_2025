%% "Vanilla" SINDy challenges and limitations
% MATLAB version of 2_SINDy_challenges.ipynb (with the MATLAB port of PySINDy, package +ps).
%
%   - Data: noise level, amount of data, sampling rate
%   - Model selection: the hyperparameter lambda (the threshold)
%   - Library: polynomials, trigonometric functions
%
% The tutorial uses the Lorenz system
%   xdot = sigma (y - x),   ydot = x (rho - z) - y,   zdot = x y - beta z
% and, for the library, an inverted pendulum.  The random numbers are MATLAB's, not numpy's, so the noisy
% models differ in the last digits from the ones printed in the notebook.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'utils'))        % lorenz_system, pendulum_ode, plot functions, setup_pysindy
setup_pysindy

% integrator tolerances as in the notebook (RK45, rtol = atol = 1e-12)
options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);

% Lorenz system parameters
sigma = 10;  rho = 28;  beta = 8/3;
lorenz_params = [sigma, rho, beta];

%% 1. Data: noise, amount, and sampling rate
% generate Lorenz data
x0 = [-8; 8; 27];
dt = 0.001;
t_final = 20;
t = (0:round(t_final / dt) - 1).' * dt;          % np.arange(0, t_final, dt)
[~, x] = ode45(@(t, x) lorenz_system(t, x, lorenz_params), t, x0, options);

% add Gaussian white noise
rng(1)
sig = 0.001;
x_noisy = x + sig * std(x(:), 1) * randn(size(x));

% build and identify the model: the library is built and the derivatives are computed inside
poly_library = ps.feature_library.PolynomialLibrary('degree', 3);
optimizer = ps.optimizers.STLSQ('threshold', 0.1);
model = ps.SINDy('feature_library', poly_library, 'optimizer', optimizer);

model.fit(x_noisy, dt, 'feature_names', {'x', 'y', 'z'});
fprintf('SINDy model with %g noise, dt = %g, t_final = %g:\n', sig, dt, t_final);
model.print()

%% 2. Model selection: sensitivity to the hyperparameter lambda
% Compare different lambdas (thresholds) to see their effect on the sparsity of the model.
threshold = 0;                          % the least-squares solution, without regularization
% threshold = 0.1;                      % a sparse solution
% threshold = 20;                       % the trivial solution (xdot = 0)

optimizer = ps.optimizers.STLSQ('threshold', threshold);
model_lambda = ps.SINDy('feature_library', poly_library, 'optimizer', optimizer);
model_lambda.fit(x_noisy, dt, 'feature_names', {'x', 'y', 'z'});
fprintf('SINDy model with threshold = %g:\n', threshold);
model_lambda.print()

%% 3. Library selection: polynomials vs. trigonometric functions
% Generate inverted pendulum data
pendulum_params = struct('m', 0.1, 'l', 0.5, 'k', 0.03, 'g', 9.8);
x0_pendulum = [0.1; 0];
dt_pendulum = 0.001;
t_final_pendulum = 20;
t_pendulum = (0:round(t_final_pendulum / dt_pendulum) - 1).' * dt_pendulum;

[~, x_pendulum] = ode45(@(t, x) pendulum_ode(t, x, pendulum_params), t_pendulum, x0_pendulum, options);

% add noise
sig_pendulum = 0.0;
x_pendulum_noisy = x_pendulum + sig_pendulum * std(x_pendulum(:), 1) * randn(size(x_pendulum));

plot_pendulum(t_pendulum, x_pendulum_noisy)

%% Case 1: polynomial library
poly_library = ps.feature_library.PolynomialLibrary('degree', 3);
optimizer = ps.optimizers.STLSQ('threshold', 0.025);
model_poly = ps.SINDy('feature_library', poly_library, 'optimizer', optimizer);
model_poly.fit(x_pendulum_noisy, dt_pendulum);

disp('SINDy model (polynomial library):')
model_poly.print()

% simulate and plot the prediction of the polynomial model
x_poly_sim = model_poly.simulate(x0_pendulum, t_pendulum);
n_sim = size(x_poly_sim, 1);                     % fewer points than t_pendulum if the model blows up
plot_sindy_pendulum(t_pendulum, x_pendulum_noisy, t_pendulum(1:n_sim), x_poly_sim)

%% Case 2: polynomial + trigonometric library
poly_library = ps.feature_library.PolynomialLibrary('degree', 3);
% the trigonometric library: sin(x), cos(x), sin(dx/dt), cos(dx/dt)
trig_library = ps.feature_library.FourierLibrary('n_frequencies', 1);
% combine the two libraries with the '+' operator
combined_library = poly_library + trig_library;

optimizer = ps.optimizers.STLSQ('threshold', 0.025);

model_trig = ps.SINDy('feature_library', combined_library, 'optimizer', optimizer);
model_trig.fit(x_pendulum_noisy, dt_pendulum);

disp('SINDy model (polynomial + trigonometric library):')
model_trig.print()

% simulate and plot the prediction of the trigonometric model
x_trig_sim = model_trig.simulate(x0_pendulum, t_pendulum);
n_sim = size(x_trig_sim, 1);
plot_sindy_pendulum(t_pendulum, x_pendulum_noisy, t_pendulum(1:n_sim), x_trig_sim)
