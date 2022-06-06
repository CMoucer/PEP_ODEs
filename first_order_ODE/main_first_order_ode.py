import numpy as np
import matplotlib.pyplot as plt

from convergence_first_order_ode import compute_convergence_guarantee

# ODE parameter
alpha = 1
# Function class
mu = 0.001  # strong-convexity parameter

# precision and verbose
epsilon = 10**-5
verbose = False

## Compute worst-case guarantee for a given Lyapunov function
# Lyapunov parameters
c = 0  # Lyapunov parameter
a = 1
tau, _, _, l = compute_convergence_guarantee(mu=mu,
                                              alpha=alpha,
                                              a=a,
                                              c=c,
                                              epsilon=epsilon,
                                              verbose=verbose)
print('Convergence guarantee for given c,a > 0, computed: ', tau, ', theory: ', 2 * mu)
print('Relative error: ', np.abs(2 * mu - tau) / (2 * mu))
print(' ')

## Compute worst-case guarantee while optimizing over Lyapunov functions
tau, c, a, _,  = compute_convergence_guarantee(mu=mu,
                                               alpha=alpha,
                                               epsilon=epsilon,
                                               verbose=verbose)
print('Convergence guarantee while optimizing over P computed: ', tau, ', theory: ', 2 * mu)
print('Relative error: ', np.abs(2 * mu - tau) / (2 * mu))
print(' ')



