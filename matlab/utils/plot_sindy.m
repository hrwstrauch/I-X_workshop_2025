function plot_sindy(t_data, x_data, t_sindy, x_sindy)
%PLOT_SINDY  Compare data with the simulation of a SINDy model of the Lorenz system (x, y, z and the attractor).
figure('Position', [100 100 560 700]);
labels = {'x', 'y', 'z'};  colors = {'b', 'r', 'g'};
for i = 1:3
    subplot(5, 1, i)
    plot(t_data, x_data(:, i), colors{i}, 'LineWidth', 1.2, 'DisplayName', ['Data ' labels{i}]), hold on
    plot(t_sindy, x_sindy(:, i), 'k--', 'LineWidth', 1, 'DisplayName', ['SINDy ' labels{i}])
    ylabel(labels{i})
    if i < 3, set(gca, 'XTickLabel', []); else, xlabel('time'); end
    legend('show')
end
subplot(5, 1, [4 5])
plot3(x_data(:, 1), x_data(:, 2), x_data(:, 3), 'b', 'LineWidth', 0.5, 'DisplayName', 'Data'), hold on
plot3(x_sindy(:, 1), x_sindy(:, 2), x_sindy(:, 3), 'k--', 'LineWidth', 0.5, 'DisplayName', 'SINDy')
xlim([-40 40]), ylim([-40 40]), zlim([-10 50])
xlabel('x'), ylabel('y'), zlabel('z'), grid on, view(3), legend('show')
end
