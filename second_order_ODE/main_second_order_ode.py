import numpy as np
import matplotlib.pyplot as plt

from convergence_second_order_ode import compute_convergence_guarantee, compute_convergence_primal

mus = np.logspace(-4, 0, 8)
L = 0
alpha = 1
a = 1
# If True, P is PSD, otherwise, the condition on P is relaxed.
psd = False

# Arrays for storing convergence rates and matrices P
taus = np.zeros(len(mus))
taus_fixed = np.zeros(len(mus))
primal_taus = np.zeros(len(mus))
Ps = np.zeros((len(mus), 2, 2)) # store the Lyapunov

for i in range(len(mus)):
    beta = 2 * np.sqrt(mus[i])
    P = np.zeros((2, 2))
    if psd:
        P[0][0] = mus[i] / 2
        P[0][1] = np.sqrt(mus[i]) / 2
        P[1][0] = np.sqrt(mus[i]) / 2
        P[1][1] = 1 / 2
    else:
        P[0][0] = 4 / 9 * mus[i]
        P[0][1] = 2 / 3 * np.sqrt(mus[i])
        P[1][0] = 2 / 3 * np.sqrt(mus[i])
        P[1][1] = 1 / 2
    taus[i], Ps[i], _, _ = compute_convergence_guarantee(mu=mus[i],
                                                 L=L,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 a=a,
                                                 P=None,
                                                 psd=psd)
    taus_fixed[i], _, _, _ = compute_convergence_guarantee(mu=mus[i],
                                                         L=L,
                                                         alpha=alpha,
                                                         beta=beta,
                                                         a=a,
                                                         P=P)
    primal_taus[i], _, _ = compute_convergence_primal(mu=mus[i],
                                                      L=L,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      P=P)

# Plot convergence guarantees
plt.plot(mus, taus, color='green', label='Optimization over Lyapunov functions')
plt.plot(mus, taus_fixed, color='orange', label='Lyapunov verification in the dual')
plt.plot(mus, -primal_taus, color='purple', label='Lyapunov verification in the primal')

if psd:
    plt.plot(mus, np.sqrt(mus), '--', color='green', label='sqrt(mu)')
else:
    plt.plot(mus, 4 / 3 * np.sqrt(mus), '--', color='green', label=' 4 / 3 sqrt(mu)')

plt.semilogy()
plt.semilogx()
plt.xlabel('strong convexity parameter')
plt.ylabel('convergence guarantee')
plt.legend()
plt.show()


## Plot Lyapunov parameters
plt.plot(mus, Ps[:, 0, 0], color='orange', label='p11')
plt.plot(mus, Ps[:, 0, 1], color='green', label='p12')
plt.plot(mus, Ps[:, 1, 1], label='p22')

if psd:
    plt.plot(mus, 1 / 2 * np.sqrt(mus), '--', color='green', label='1/2 sqrt(mu)')
    plt.plot(mus, 1 / 2 * mus, '--', color='orange', label='1/2 mu')
else:
    plt.plot(mus, 2/3 * np.sqrt(mus), '--', color='green', label='2/3 sqrt(mu)')
    plt.plot(mus, 4/9 * mus, '--', color='orange', label='4/9 mu')

plt.semilogy()
plt.semilogx()
plt.xlabel('strong convexity parameter')
plt.ylabel('convergence guarantee')
plt.legend()
plt.show()


