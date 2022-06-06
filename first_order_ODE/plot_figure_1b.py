import numpy as np
import matplotlib.pyplot as plt

from worst_case_function import worst_case_gf, compute_worst_case_primal

# number of points per dimension
nn = 100
X = np.linspace(-10, 10, nn)
mu = 0.1
L = 0
b = 0
wc, G, F = compute_worst_case_primal(mu, L, b, a=1, alpha=1)

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
