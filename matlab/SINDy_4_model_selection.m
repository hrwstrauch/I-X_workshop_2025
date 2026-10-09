%% SINDy model selection: how to select / tune the hyperparameters
% MATLAB version of 4_SINDy_model_selection.ipynb (with the MATLAB port of PySINDy, package +ps).
%
%   - k-fold cross-validation:      Brunton et al. 2016   https://www.pnas.org/doi/abs/10.1073/pnas.1517384113
%   - Akaike information criterion: Mangan et al. 2017    https://royalsocietypublishing.org/doi/full/10.1098/rspa.2017.0009
%   - Stability selection:          Maddu et al. 2022     https://royalsocietypublishing.org/doi/full/10.1098/rspa.2021.0916
%
% The tutorial uses the Lorenz system.  The random numbers (noise, the folds, the subsamples, the initial
% conditions) are MATLAB's, not numpy's, so the numbers differ in detail from the notebook; the pictures
% and the selected models should look alike.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'utils'))
setup_pysindy

% ignore warnings during the sweeps over lambda, e.g. "all coefficients removed = trivial solution"
warning('off', 'ps:stlsq')
warning('off', 'MATLAB:ode113:IntegrationTolNotMet')

%% 1. Generate data: the Lorenz system
x0 = [-8; 8; 27];
dt = 0.001;
t_final = 20;
t = (0:round(t_final / dt) - 1).' * dt;
sigma = 10;  rho = 28;  beta = 8/3;
lorenz_params = [sigma, rho, beta];
options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
[~, x] = ode45(@(t, x) lorenz_system(t, x, lorenz_params), t, x0, options);

% add Gaussian white noise
rng(1)
sig = 0.01;
x_noisy = x + sig * std(x(:), 1) * randn(size(x));

%% SINDy path: sweep over lambda, starting from the least-squares solution (lambda = 0)
% the library is built and the derivatives are computed inside the model
poly_library = ps.feature_library.PolynomialLibrary('degree', 3);

lambda_path = logspace(-3, 1, 50);
nonzero_terms = zeros(size(lambda_path));

for i = 1:numel(lambda_path)
    optimizer = ps.optimizers.STLSQ('threshold', lambda_path(i));
    model = ps.SINDy('feature_library', poly_library, 'optimizer', optimizer);
    model.fit(x_noisy, dt, 'feature_names', {'x', 'y', 'z'});
    nonzero_terms(i) = nnz(model.coefficients());       % number of non-zero terms
end

% plot: number of active terms vs. lambda
figure('Position', [100 100 700 300])
semilogx(lambda_path, nonzero_terms, 'k-', 'DisplayName', 'SINDy path'), hold on
nat = 7;                                % the Lorenz system has 7 non-zero terms (3 in xdot, 2 in ydot, 2 in zdot)
semilogx(lambda_path([1 end]), [nat nat], 'b--', 'DisplayName', 'Lorenz system number of terms')
xlabel('Threshold \lambda'), ylabel('Number of active terms')
legend('show'), title('Number of active terms vs. \lambda')

%% 2. k-fold cross-validation
% pre-compute the time derivatives (fourth-order central differences along the time axis)
x_dot_noisy = ps.differentiation.FiniteDifference('axis', 1, 'order', 4).call(x_noisy, t);

% the SINDy library
library = ps.feature_library.PolynomialLibrary('degree', 3);

% cross-validation to find the best threshold
thresholds = logspace(-3, 1, 15);       % range of regularization thresholds
k = 10;                                 % number of folds
N = size(x_noisy, 1);

% k folds of a random permutation of the samples (like sklearn's KFold with shuffle = True)
rng(42)
perm = randperm(N);
fold_size = floor(N / k) * ones(1, k);
fold_size(1:mod(N, k)) = fold_size(1:mod(N, k)) + 1;
bounds = [0, cumsum(fold_size)];

mse_scores = zeros(numel(thresholds), k);
mean_squared_error = @(y_true, y_pred) mean((y_true - y_pred).^2, 'all');

for i = 1:numel(thresholds)
    % initialize the SINDy model for this threshold
    model = ps.SINDy('optimizer', ps.optimizers.STLSQ('threshold', thresholds(i)), 'feature_library', library);
    % go through each fold
    for j = 1:k
        val_idx = sort(perm(bounds(j) + 1:bounds(j + 1)));
        train_idx = setdiff(1:N, val_idx);
        % fit on the training data
        model.fit(x_noisy(train_idx, :), t(train_idx), 'x_dot', x_dot_noisy(train_idx, :));
        % mean squared error on the validation data, with model.score()
        mse_scores(i, j) = model.score(x_noisy(val_idx, :), t(val_idx), 'x_dot', x_dot_noisy(val_idx, :), ...
            'metric', mean_squared_error);
    end
end

% find the optimal model
mean_mse = mean(mse_scores, 2);
[~, best_threshold_idx] = min(mean_mse);
best_threshold = thresholds(best_threshold_idx);

fprintf('Optimal threshold found via cross-validation: %.4f\n', best_threshold);

% plot the results
figure('Position', [100 100 600 400])
semilogx(thresholds, mean_mse, 'b-', 'DisplayName', 'Mean normalized MSE'), hold on
scatter(thresholds(best_threshold_idx), mean_mse(best_threshold_idx), 100, 'r', 'filled', 'DisplayName', 'Optimal threshold')
xlabel('Regularization threshold \lambda'), ylabel('Normalized mean squared error (MSE)')
title('Cross-validation for SINDy model selection'), legend('show'), grid on

% train the final model on the full data set with the optimal threshold
final_model = ps.SINDy('optimizer', ps.optimizers.STLSQ('threshold', best_threshold), 'feature_library', library);
final_model.fit(x_noisy, t, 'x_dot', x_dot_noisy);

fprintf('\nFinal SINDy model with the optimal threshold:\n');
final_model.print()

%% 3. Prediction error and sparsity: Akaike information criterion
% range of thresholds: the number of lambdas (models of different complexity) to sweep through
nl_aic = 15;
lambda_aic = logspace(-3, 1, nl_aic);

% prediction error: nm initial conditions chosen at random from the time series
nm = 10;                                % number of initial conditions for the prediction
tF = 0.2;                               % time span of the model prediction
tspanE = (0:round(tF / dt) - 1).' * dt;
ltE = numel(tspanE);                    % time steps of the prediction
lt = size(x, 1);                        % length of the time series

rng(1)
x0i = randperm(lt - ltE, nm);           % random selection of the initial conditions (without replacement)
x0e = x(x0i, :);                        % the (noise-free) initial states

k_aic = zeros(1, nl_aic);
logL_aic = zeros(1, nl_aic);
aic_score = zeros(1, nl_aic);
mse_sim = zeros(nl_aic, nm);

% sweep over the lambda path
for i = 1:nl_aic
    % fit the SINDy model
    optimizer = ps.optimizers.STLSQ('threshold', lambda_aic(i));
    model = ps.SINDy('feature_library', poly_library, 'optimizer', optimizer);
    model.fit(x_noisy, dt, 'feature_names', {'x', 'y', 'z'});

    % run the model predictions from the different initial conditions
    for j = 1:nm
        % integrate the SINDy model; a model that blows up (or whose integration fails) gets the penalty 1e4
        try
            x_test_sim = model.simulate(x0e(j, :), tspanE);
            blown_up = size(x_test_sim, 1) ~= ltE || any(~isfinite(x_test_sim(:))) || any(x_test_sim(:) > 1e4);
        catch
            blown_up = true;
        end
        if blown_up
            x_test_sim = 1e4;
        end
        x_test = x(x0i(j):x0i(j) + ltE - 1, :);
        % model prediction error
        mse_sim(i, j) = sum((x_test - x_test_sim).^2, 'all');
    end

    k_aic(i) = nnz(model.coefficients());                          % number of non-zero terms: the sparsity
    logL_aic(i) = nm * log(mean(mse_sim(i, :).^2));                % log of the mean prediction error
    aic_score(i) = logL_aic(i) + 2 * k_aic(i);
end

% find the minimum score
[minAIC, min_AIC_idx] = min(aic_score);
lambda_AIC_min = lambda_aic(min_AIC_idx);

% train the final model on the full data set with the optimal threshold
final_model = ps.SINDy('optimizer', ps.optimizers.STLSQ('threshold', lambda_AIC_min), 'feature_library', library);
final_model.fit(x_noisy, t, 'x_dot', x_dot_noisy);

fprintf('\nFinal SINDy model with the AIC threshold %.4f:\n', lambda_AIC_min);
final_model.print()

% plot the scores, the prediction error and the 2k curve
figure('Position', [100 100 600 400])
semilogx(lambda_aic, aic_score - minAIC, 'LineWidth', 1.4, 'DisplayName', 'AIC'), hold on
semilogx(lambda_aic, logL_aic - minAIC, 'DisplayName', 'Model prediction error: logL')
semilogx(lambda_aic, 2 * k_aic, 'DisplayName', 'Model complexity: 2k')
% a marker at the selected model (minimum AIC)
plot(lambda_aic(min_AIC_idx), aic_score(min_AIC_idx) - minAIC, 'r*', 'MarkerSize', 10, 'DisplayName', 'Selected model')
ylabel('Score'), xlabel('Regularization threshold \lambda'), legend('show')

%% 4. Stability selection
% less noise, to see the method better
rng(1)
sig = 0.001;
x_noisy = x + sig * std(x(:), 1) * randn(size(x));

% compute the derivative
x_dot_noisy = ps.differentiation.FiniteDifference('axis', 1, 'order', 4).call(x_noisy, t);

% the library
polyorder = 3;
n = 3;                                  % number of states
poly_library = ps.feature_library.PolynomialLibrary('degree', polyorder, 'include_bias', true);

% the number of features: the library has to be fitted to a data set first
poly_library.fit(x_noisy);
m = poly_library.n_output_features_;

% the number of lambdas (models of different complexity) to sweep through
nl = 15;
lambda_list = logspace(1, -3, nl);
nT = floor(size(x_noisy, 1) / 2);       % size of the subsamples: half the number of measurements
B = 50;                                 % number of subsampled models (Maddu et al. recommend 250)
rng(1)

coeffs_active = false(n, m, nl, B);

% subsample the data
for j = 1:B
    indices = randperm(2 * nT, nT);                          % a random half of the samples

    % for every subsample: sweep over the model complexity and store the structure of the model
    for i = 1:nl
        optimizer = ps.optimizers.STLSQ('threshold', lambda_list(i));
        model = ps.SINDy('feature_library', poly_library, 'optimizer', optimizer);
        model.fit(x_noisy(indices, :), dt, 'x_dot', x_dot_noisy(indices, :));
        coeffs_active(:, :, i, j) = model.coefficients() ~= 0;     % active terms (logical)
    end
end

%% Plot the stability paths
sStable = 0.8;                          % the "stable inclusion probability"
states = {'x', 'y', 'z'};

% mean inclusion probability over the subsamples
XiEM = mean(coeffs_active, 4);

% the "true active terms" of the Lorenz system, by hand (the library: 1, x, y, z, xx, xy, xz, ...)
active = false(m, n);
active([2 3], 1) = true;                % terms 2 and 3 for state 1 (x)
active([2 3 7], 2) = true;              % terms 2, 3 and 7 for state 2 (y)
active([4 6], 3) = true;                % terms 4 and 6 for state 3 (z)

figure('Position', [100 100 1100 380])
for i = 1:n
    subplot(1, 3, i)
    Y_on = reshape(XiEM(i, active(:, i), :), nnz(active(:, i)), nl);
    Y_off = reshape(XiEM(i, ~active(:, i), :), nnz(~active(:, i)), nl);
    h_off = semilogx(lambda_list, Y_off.', 'k--'); hold on
    h_on = semilogx(lambda_list, Y_on.', 'b');
    yline(sStable, 'r-');
    ylim([0 1.1]), grid on
    title(states{i}), xlabel('Regularization threshold \lambda')
    if i == 1, ylabel('Stability, or importance measure'); end
    if i == 3, legend([h_on(1), h_off(1)], {'True active terms', 'Non-active terms'}, 'Location', 'northeast'); end
end

%% Select a model from the stability paths
% visual inspection of the paths: a large gap around lambda = 5e-1
lambdaSS = 5e-1;

% train the final model on the full data set
final_model = ps.SINDy('optimizer', ps.optimizers.STLSQ('threshold', lambdaSS), 'feature_library', library);
final_model.fit(x_noisy, t, 'x_dot', x_dot_noisy);

fprintf('\nFinal SINDy model with the stability-selection threshold:\n');
final_model.print()

warning('on', 'ps:stlsq')
warning('on', 'MATLAB:ode113:IntegrationTolNotMet')
