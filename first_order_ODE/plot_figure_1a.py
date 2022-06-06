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
saved = False
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