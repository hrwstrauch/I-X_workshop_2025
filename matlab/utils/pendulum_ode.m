function dx = pendulum_ode(~, x, p)
%PENDULUM_ODE  Inverted pendulum with friction, x = [angle; angular velocity].
%   p is a struct with the mass m, the length l, the friction coefficient k and gravity g.
dx = [x(2);
      (p.g / p.l) * sin(x(1)) - (p.k / (p.m * p.l^2)) * x(2)];
end
