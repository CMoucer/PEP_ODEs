import numpy as np
import matplotlib.pyplot as plt

from convergence_first_order_ode import compute_convergence_guarantee
from worst_case_function import worst_case_gf, compute_worst_case_primal

# Set the ODE : dx/dt = -alpha grad(f)(x)
alpha = 1
# Set the function parameters
mus = np.logspace(-3, 0, 15)
L = 0
a = 1 # scalar

### Compute
## Given a Lyapunov function V(x(t)) = a* f(x(t)) - f_* + P * ||x(t) - x_*||^2, compute worst-case guarantee
#P = 0 # scalar
P = None
taus = np.zeros(len(mus))

for i in range(len(mus)):
    taus[i], _, _ = compute_convergence_guarantee(mu=mus[i],
                                            L=L,
                                            alpha=alpha,
                                            a=a,
                                            P=P)

if P== None:
    plt.plot(mus, taus, label='PEP optimized over Lyapunov functions')
else:
    plt.plot(mus, taus, label='PEP for a given Lyapunov function')
plt.plot(mus, 2 * mus, label='Theoretical guarantee')
plt.semilogx()
plt.semilogy()
plt.ylabel('worst-case convergence guarantee')
plt.xlabel('strong convexity parameter')
plt.legend()
plt.show()


# WORST CASE FUNCTION
plot_worst_case = False
if plot_worst_case:
    # number of points per dimension
    nn = 100
    X = np.linspace(-10, 10, nn)
    mu = 0.1
    L = 0
    b = 0
    wc, G, F = compute_worst_case_primal(mu, L, b, a=1, alpha = 1)

    # Evaluate the function on a grid
    sol = np.zeros(nn)
    for i in range(nn):
        xx = np.array([X[i]])
        sol[i], _, _, _ = worst_case_gf(L, mu, G, F, xx, N=1)

        # Plot the function
    plt.plot(X, sol, label='worst-case function')
    plt.plot(X, mu / 2 * X ** 2, label='mu/2*x^2')
    plt.xlabel('x')
    plt.ylabel('worst-case function')
    plt.legend()
    plt.show()


