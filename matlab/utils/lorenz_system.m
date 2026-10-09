function dx = lorenz_system(~, x, params)
%LORENZ_SYSTEM  Right-hand side of the Lorenz system, params = [sigma, rho, beta].
dx = [params(1) * (x(2) - x(1));
      x(1) * (params(2) - x(3)) - x(2);
      x(1) * x(2) - params(3) * x(3)];
end
