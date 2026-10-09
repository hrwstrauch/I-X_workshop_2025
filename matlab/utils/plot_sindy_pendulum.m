function plot_sindy_pendulum(t_data, x_data, t_sindy, x_sindy)
%PLOT_SINDY_PENDULUM  Compare pendulum data with the simulation of a SINDy model.
figure('Position', [100 100 560 420]);
subplot(2, 1, 1)
plot(t_data, x_data(:, 1), 'b', 'LineWidth', 1.2, 'DisplayName', 'Data x'), hold on
plot(t_sindy, x_sindy(:, 1), 'k--', 'LineWidth', 1, 'DisplayName', 'SINDy x')
ylabel('x'), set(gca, 'XTickLabel', []), legend('show')
subplot(2, 1, 2)
plot(t_data, x_data(:, 2), 'r', 'LineWidth', 1.2, 'DisplayName', 'Data dx/dt'), hold on
plot(t_sindy, x_sindy(:, 2), 'k--', 'LineWidth', 1, 'DisplayName', 'SINDy dx/dt')
ylabel('dx/dt'), xlabel('time'), legend('show')
end
