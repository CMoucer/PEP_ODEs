import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def compute_worst_case_primal(mu, L, b, a=1, alpha = 1):
    """
    Consider the gradient flow,
            d/dt x(t) = - alpha * grad(f(x(t));
    where f is a L-smooth, mu-strongly convex function, and alpha a positive parameter.

    We study the convergence of a quadratic Lyapunov function
            V(x(t)) = a(f(x(t)) - f^*) + b||x(t)-x^*||^2,
    where a, b are positive parameters.

    This code aims to compute the worst-case convergence rate tau(mu, L, a, b, alpha), such that,
            d/dt V(x(t)) <= - tau(mu, L, a, b, alpha) V(x(t)),
    for all L-smooth mu-strongly convex functions, and all trajectories x(t) generated by the gradient flow.

    :param mu: strong convexity parameter
    :param L: smoothness parameter
    :param a: Lyapunov parameter (positive)
    :param b: Lyapunov parameter (positive)
    :param alpha: ODE parameter
    :return:
        - worst-case tau(mu, L, a, b, alpha)
        - Gram matrix G (primal formulation)
        - F (primal formulation)
    """
    ## PARAMETERS
    npt = 2  # optimal and initial point
    dimF = 1  # one dimension only
    dimG = 2  # (x(t) // grad f(x(t)))

    ## INITIALIZE
    FF, GG, XX = [], [], []
    # initial point
    y = np.zeros((1, dimG))
    y[0][0] = 1.
    XX.append(y)
    # construction of the base
    f, g = np.zeros((1, dimF)), np.zeros((1, dimG))
    f[0][0], g[0][1] = 1., 1.
    FF.append(f)
    GG.append(g)
    # optimal point : equal to zero
    XX.append(np.zeros((1, dimG)))
    FF.append(np.zeros((1, dimF)))
    GG.append(np.zeros((1, dimG)))

    ## PEP FORMULATION
    # VARIABLES
    G = cp.Variable((dimG, dimG), symmetric=True) # Gram matrix
    F = cp.Variable((dimF, 1)) # function values

    # Compute the Lyapunov and its derivative
    X_t = XX[0][0]
    F_t = FF[0]
    G_t = GG[0][0]
    X_dot = - alpha * G_t
    lyap = a * (F_t @ F[:, 0])[0] + b * X_t @ G @ X_t
    lyap_dot = a * X_dot @ G @ G_t + b * (X_t @ G @ X_dot + X_dot @ G @ X_t)

    # CONSTRAINTS
    ### Positivity of the Gram matrix
    constraints = [G >> 0.]
    ### Constraint on the previous iterate :
    constraints = constraints + [lyap == 1.]
    ### Interpolation inequalities
    for i in range(npt):
        for j in range(npt):
            if j != i:
                if (mu != 0 & L != 0):
                    A = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                        1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                                mu * np.dot((XX[i] - XX[j]).T, XX[i] - XX[j]) -
                                                2 * mu / L * np.dot((XX[i] - XX[j]).T, GG[i] - GG[j]))
                if (mu != 0 & L == 0):
                    A = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                        1 / 2 * mu * np.dot((XX[i] - XX[j]).T, XX[i] - XX[j])
                if (mu == 0 & L != 0):
                    A = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                        1 / 2 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j])
                if (mu == 0 & L == 0):
                    A = np.dot((XX[i] - XX[j]).T, GG[j])
                A = .5 * (A + A.T)
                b = FF[j] - FF[i]
                constraints += [b[0] @ F[:, 0] + sum([(A @ G)[k, k] for k in range(dimG)]) <= 0.]

    ## OPTIMIZE
    prob = cp.Problem(cp.Maximize(lyap_dot), constraints)
    prob.solve(cp.SCS)

    return prob.value, G.value, F.value


def worst_case_gf(L, mu, G, F, x, N=1, eps=0):
    """
    :param L: smoothness parameter
    :param mu: strong convexity parameter
    :param G: Gram matrix obtained from PEP estimation
    :param F: Function values obtained from PEP estimation
    :param x: new points where f is evaluated
    :param N: dimension
    :param eps: precision
    :return:
    """
    eigenvalues, eigenvector = np.linalg.eig(G)
    eigs = np.zeros(len(eigenvalues))
    for i in range(N):
        eigs[i] = eigenvalues[i]
    new_D = np.diag(np.sqrt(eigs))
    Q, R = np.linalg.qr(np.dot(new_D, eigenvector.T))
    # Keep the principal dimensions
    R = R[:N, :]
    # Reconstruction of x_i based on the approximating R
    XX = [R[:, 0]]  # initial points
    GG = []
    for i in range(1, R.shape[1]):
        GG.append(R[:, i])  # basis

    ## INTERPOLATION OF THE FUNCTION (QCQP PROBLEM)
    m1 = np.array([[-mu * L, mu * L], [mu * L, -mu * L]])
    m2 = np.array([[mu, -L], [-mu, L]])
    m3 = np.array([[-1, 1], [1, -1]])
    M1 = np.kron(m1, np.eye(R.shape[0]))
    M2 = np.kron(m2, np.eye(R.shape[0]))
    M3 = np.kron(m3, np.eye(R.shape[0]))
    f = cp.Variable(1)
    g = cp.Variable(R.shape[0])
    constraints = [f >= -1.]

    X = np.array([x[0], XX[0][0]]).T
    G = np.array([0, GG[0][0]]).T
    U1 = np.zeros((1, 2))
    U1[0][0] = 1.
    G = G + g @ U1
    X_ = np.array([XX[0][0], x[0]]).T
    G_ = np.array([GG[0][0], 0]).T
    U2 = np.zeros((1, 2))
    U2[0][1] = 1.
    G_ = G_ + g @ U2

    # add constraints
    constraints = constraints + [
        (L - mu) * (f - F[0][0]) + 1 / 2 * (X.T @ M1 @ X + 2 * (X.T @ M2 @ G) + cp.quad_form(G, M3)) >= -eps]
    constraints = constraints + [
        (L - mu) * (F[0][0] - f) + 1 / 2 * (X_.T @ M1 @ X_ + 2 * (X_.T @ M2 @ G_) + cp.quad_form(G_, M3)) >= -eps]

    # formulate the problem
    prob = cp.Problem(cp.Maximize(f), constraints)
    prob.solve(cp.SCS)
    # print('The interpolating function evaluated at point {} is equal to'.format(str(x)), prob.value)

    return prob.value, XX, GG, g.value