clear all
close all
clc

alpha = 0;
a =@(x) 4*(2*norm(x-0.5))^alpha;   % stiffness
u0 = @(x) sin(pi*x/2);             % initial condition

L = 1;
N = 100;
x = linspace(0,L);
dx = x(2) - x(1);

A = sparse(N,N);
A(1,1) = -2*a(x(1)); A(1,2) = 2*a(x(1));
for ii = 2:N-1
  A(ii,ii-1) =  1*a(x(ii));
  A(ii,ii)   = -2*a(x(ii));
  A(ii,ii+1) =  1*a(x(ii));
end
A(N,N-1) = 2*a(x(N)); A(N,N) = -2*a(x(N));
A = A/dx^2;

T = 2;
NT = 400;
time = linspace(0,T,NT);
dt = time(2) - time(1);

U = zeros(N,NT);
U(:,1) = u0(x);                  % zero velocity initial condition
U(:,2) = u0(x);

cdofs = [1];                     % nodes with Dirichlet BCs
fdofs = setdiff(1:N, cdofs);

for ii = 3:NT
  U(fdofs,ii) = 2*U(fdofs,ii-1) - U(fdofs,ii-2) + dt^2*A(fdofs,fdofs)*U(fdofs,ii-1);
end

figure
surf(time,fliplr(x),U)
xlabel 'time'
ylabel 'x'
ax = gca;
ax.YTick = [0, 0.5, 1];
ax.YTickLabel = {'1', '0.5', '0'};
zlabel 'solution'
shading interp
