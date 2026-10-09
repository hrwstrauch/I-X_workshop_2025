function plot_pendulum(t, x, name)
%PLOT_PENDULUM  Time series of the inverted pendulum.
if nargin < 3, name = 'Pendulum Time Series Data'; end
figure('Position', [100 100 560 420]);
subplot(2, 1, 1), plot(t, x(:, 1), 'b'), ylabel('x'), set(gca, 'XTickLabel', []), title(name)
subplot(2, 1, 2), plot(t, x(:, 2), 'r'), ylabel('dx/dt'), xlabel('time')
end
