%% Ensemble SINDy
% MATLAB version of 5_Ensemble-SINDy.ipynb (with the MATLAB port of PySINDy, package +ps).
%
%   - Robust model discovery in the high-noise and low-data limit
%   - Uncertainty quantification
%
% The tutorial uses the Lorenz system.  E-SINDy fits many SINDy models, each to a bootstrap sample of the data
% (bagging), and takes the median of their coefficients: a term that appears in less than half of the models
% is removed, and the spread of the models is an uncertainty estimate.
% The random numbers are MATLAB's, not numpy's, so the table and the histograms differ in detail from the
% notebook; the trend (E-SINDy has fewer wrong terms and a smaller error than SINDy) is the same.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'utils'))
setup_pysindy

warning('off', 'ps:stlsq')

%% 1. Generate data: the Lorenz system
x0 = [-8; 8; 27];
dt = 0.01;
t_final = 10;
t = (0:round(t_final / dt) - 1).' * dt;
sigma = 10;  rho = 28;  beta = 8/3;
lorenz_params = [sigma, rho, beta];
options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
[~, x_clean] = ode45(@(t, x) lorenz_system(t, x, lorenz_params), t, x0, options);

% add Gaussian white noise
rng(1)
sig = 0.01;
x_noisy = x_clean + sig * std(x_clean(:), 1) * randn(size(x_clean));  % (as in the notebook; not used below)

%% 2. SINDy and E-SINDy as functions of the noise level
% ensemble and SINDy hyperparameters
n_ensembles = 100;
lambda_val = 0.2;                       % the SINDy sparsification knob

% noise levels
sig_nn = 0:0.01:0.05;
n_noise = numel(sig_nn);
m_noise = 10;                           % number of noise realizations

% output arrays
nWrongTerms = zeros(n_noise, m_noise);
modelError = zeros(n_noise, m_noise);
nWrongTermsE = zeros(n_noise, m_noise);
modelErrorE = zeros(n_noise, m_noise);

poly_library = ps.feature_library.PolynomialLibrary('degree', 3, 'include_bias', true);
poly_library.fit(x_clean);
features = poly_library.get_feature_names();
n = 3;                                  % number of states
m = numel(features);                    % number of library features

feature_names = {'x', 'y', 'z'};

% the true coefficients: one row per library term, one column per state
true_coeffs = zeros(m, n);
true_coeffs(2, 1) = -10;                % state 1 (x)
true_coeffs(3, 1) = 10;
true_coeffs(2, 2) = 28;                 % state 2 (y)
true_coeffs(3, 2) = -1;
true_coeffs(7, 2) = -1;
true_coeffs(4, 3) = -8/3;               % state 3 (z)
true_coeffs(6, 3) = 1;

% loop over the noise realizations and levels
for mm = 1:m_noise
    for nn = 1:n_noise

        % add noise (seed mm - 1, as np.random.seed(mm) with mm = 0, 1, ...)
        rng(mm - 1)
        x = x_clean + sig_nn(nn) * std(x_clean(:), 1) * randn(size(x_clean));

        % derivatives by a fourth-order finite difference
        differentiation_method = ps.differentiation.FiniteDifference('order', 4);

        % the library of terms
        poly_library = ps.feature_library.PolynomialLibrary('degree', 3, 'include_bias', true);

        % fit the SINDy model
        sindy_model = ps.SINDy('feature_library', poly_library, ...
            'optimizer', ps.optimizers.STLSQ('threshold', lambda_val), ...
            'differentiation_method', differentiation_method);
        sindy_model.fit(x, dt);
        sindy_coeffs = sindy_model.coefficients();

        % fit the E-SINDy model: bagging
        ensemble_optimizer = ps.optimizers.EnsembleOptimizer(ps.optimizers.STLSQ('threshold', lambda_val), ...
            'bagging', true, 'n_models', n_ensembles);
        esindy_model = ps.SINDy('feature_library', poly_library, ...
            'optimizer', ensemble_optimizer, ...
            'differentiation_method', differentiation_method);
        esindy_model.fit(x, dt, 'feature_names', feature_names);

        % the coefficients of all the models: n_states x n_terms x n_ensembles
        ensemble_coefs = ensemble_optimizer.coef_list;
        median_ensemble = median(ensemble_coefs, 3);
        esindy_coeffs = median_ensemble;

        % compare SINDy and E-SINDy: SINDy
        sindy_active = sindy_coeffs ~= 0;
        true_active = true_coeffs ~= 0;
        nWrongTerms(nn, mm) = sum(sindy_active ~= true_active.', 'all');
        modelError(nn, mm) = norm(sindy_coeffs - true_coeffs.', 'fro') / norm(true_coeffs, 'fro');

        % E-SINDy
        esindy_active = esindy_coeffs ~= 0;
        nWrongTermsE(nn, mm) = sum(esindy_active ~= true_active.', 'all');
        modelErrorE(nn, mm) = norm(esindy_coeffs - true_coeffs.', 'fro') / norm(true_coeffs, 'fro');
    end
end

% average over the noise realizations
avg_nWrongTerms = mean(nWrongTerms, 2);
avg_modelError = mean(modelError, 2);
avg_nWrongTermsE = mean(nWrongTermsE, 2);
avg_modelErrorE = mean(modelErrorE, 2);

disp('Comparison results:')
disp('Noise level | SINDy wrong terms | E-SINDy wrong terms | SINDy error | E-SINDy error')
disp('-----------------------------------------------------------------------------------')
for i = 1:n_noise
    fprintf('%8.2f    | %11.2f       | %13.2f       | %8.2f    | %8.2f\n', ...
        sig_nn(i), avg_nWrongTerms(i), avg_nWrongTermsE(i), avg_modelError(i), avg_modelErrorE(i));
end

% plot the results
figure('Position', [100 100 700 560])
subplot(2, 1, 1)
plot(sig_nn, avg_nWrongTerms, 'DisplayName', 'SINDy'), hold on
plot(sig_nn, avg_nWrongTermsE, 'DisplayName', 'E-SINDy')
xlabel('Noise level'), ylabel('Number of wrong terms')
legend('Location', 'northwest'), title('Wrong terms vs. noise level'), grid on
subplot(2, 1, 2)
plot(sig_nn, avg_modelError, 'DisplayName', 'SINDy'), hold on
plot(sig_nn, avg_modelErrorE, 'DisplayName', 'E-SINDy')
xlabel('Noise level'), ylabel('Model coefficient error')
title('Model coefficient error vs. noise level'), grid on

%% 3. The distribution of every coefficient over the ensemble (last fit: the largest noise level)
figure('Position', [100 100 700 1000])
min_overall = -15;
max_overall = 30;

xticknames = esindy_model.get_feature_names();
ensemble_coefs = ensemble_optimizer.coef_list;          % n_states x n_terms x n_ensembles
bins = linspace(min_overall, max_overall, max_overall - min_overall + 1);

% the first 10 library terms (1, x, y, z, xx, xy, xz, yy, yz, zz) of each of the 3 states
for j = 1:10
    for i = 1:3
        subplot(10, 3, i + (j - 1) * 3)
        histogram(squeeze(ensemble_coefs(i, j, :)), 'BinEdges', bins, 'FaceColor', 'b'), hold on
        if true_coeffs(j, i) ~= 0
            plot(true_coeffs(j, i), 0, 'rX', 'MarkerSize', 10, 'LineWidth', 2)
        else
            plot(true_coeffs(j, i), 0, 'kX', 'MarkerSize', 10, 'LineWidth', 2)
        end
        grid on
        if j == 1, title(feature_names{i}); end
        if i == 1, ylabel(xticknames{j}); else, set(gca, 'YTickLabel', []); end
        if j ~= 10, set(gca, 'XTickLabel', []); else, xlabel('coefficient value'); end
        xticks([min_overall, 0, floor(max_overall / 2), max_overall])
        xlim([min_overall, max_overall])
    end
end

warning('on', 'ps:stlsq')
