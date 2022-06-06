import numpy as np
import matplotlib.pyplot as plt

from convergence_second_order_ode import compute_convergence_guarantee, compute_convergence_primal

# Function parameter
mu = 0.01 # strong convexity
# ODE parameters
alpha = 1
beta = 2 * np.sqrt(mu)
# Precision and verbose
epsilon = 10**-4
verbose = False
# Relaxation on P or not
psd = False

## COMPUTE WORST-CASE GUARANTEE WHILE OPTIMIZING OVER P
relaxed_tau, relaxed_P, _, nu, a = compute_convergence_guarantee(mu=mu,
                                                                 alpha=alpha,
                                                                 beta=beta,
                                                                 psd=psd,
                                                                 epsilon=epsilon,
                                                                 verbose=verbose)
if not psd:
    print('Convergence guarantee while optimizing over P', relaxed_tau, 4/3 * np.sqrt(mu))
else:
    print('Convergence guarantee while optimizing over P', relaxed_tau, np.sqrt(mu))

## VERIFY WORST-CASE GUARANTEE FOR A GIVEN LYAPUNOV
P = np.zeros((2, 2))
a = 1

if not psd:
    P[0][0] = 4 / 9 * mu
    P[1][0] = 2 / 3 * np.sqrt(mu)
    P[0][1] = 2 / 3 * np.sqrt(mu)
    P[1][1] = 1 / 2
else:
    P[0][0] = mu / 2
    P[1][0] = np.sqrt(mu) / 2
    P[0][1] = np.sqrt(mu) / 2
    P[1][1] = 1 / 2

tau_verif, _, _, _, _ = compute_convergence_guarantee(mu=mu,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       a=a,
                                                       P=P,
                                                       epsilon=epsilon,
                                                       verbose=verbose)


## COMPARE IN THE PRIMAL
primal_tau, _, _ = compute_convergence_primal(mu=mu,
                                              L=0,
                                              alpha=alpha,
                                              beta=beta,
                                              P=P)
if not psd:
    print('Convergence guarantee for fixed P (relaxed), computed: ', tau_verif, ', theory: ', 4 / 3 * np.sqrt(mu))
    print('Primal : convergence guarantee for fixed P (relaxed), computed: ', -primal_tau, ', theory: ', 4 / 3 * np.sqrt(mu))
else:
    print('Convergence guarantee for fixed P (psd)', tau_verif, np.sqrt(mu))
    print('Primal : convergence guarantee for fixed P (psd) ', -primal_tau, np.sqrt(mu))




