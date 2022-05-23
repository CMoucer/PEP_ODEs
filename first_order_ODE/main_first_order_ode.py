import numpy as np
import matplotlib.pyplot as plt

from convergence_first_order_ode import compute_convergence_guarantee
from worst_case_function import worst_case_gf, compute_worst_case_primal

### PARAMETERS
# Set the ODE : dx/dt = -alpha grad(f)(x)
alpha = 1
# Set the function parameters
mus = np.logspace(-4, 0, 15) # strong convexity parameter
a = 1 # scalar
# precision and verbose
epsilon = 10**-5
verbose = False


### COMPUTE WORST-CASE GUARANTEE
## Given a Lyapunov function V(x(t)) = a* f(x(t)) - f_* + c * ||x(t) - x_*||^2, compute worst-case guarantee
#c = 0 # scalar
c = None
taus = np.zeros(len(mus))

for i in range(len(mus)):
    taus[i], _, _, _ = compute_convergence_guarantee(mu=mus[i],
                                                  alpha=alpha,
                                                  a=a,
                                                  c=c,
                                                  epsilon=epsilon,
                                                  verbose=verbose)

### PLOT
## Plot convergence guarantee
if c== None:
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

# Save convergence guarantee
saved = True
if saved:
    saved_txt = np.array([mus, taus, 2*mus])
    np.savetxt('/Users/cmoucer/PycharmProjects/ContinuousPEP/output/gradient_flow/gf.txt',
               saved_txt.T,
               delimiter=' ',
               header="condition gf theorygf")

## Plot relative scale error
relative_error = False
if relative_error:
    plt.plot(mus, np.abs(2*mus - taus)/(2*mus))
    plt.semilogx()
    plt.xlabel('strong convexity parameter')
    plt.ylabel('relative scale error')
    plt.show()


## Plot function in the worst-case
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


