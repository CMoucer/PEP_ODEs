import numpy as np
import cvxpy as cp

def compute_convergence_guarantee(mu, L=0, alpha=1, a=1., P = None,  upper=2., lower=0., epsilon=10 ** -4):
        """
        Consider the gradient flow
                dx/dt = - alpha * grad f(x(t)),
        where alpha is a positive parameter, f a L-smooth, mu-strongly convex function, and x* a stationary point.

        We consider quadratic Lyapunov functions
                V(x(t)) = a(f(x(t)) - f^*) + P ||x(t) - x_*||^2,
        where a,c are positive parameters.

        This code compute the worst-case guarantee tau(mu, L, alpha, P) that verifies
                d/dt V(x) <= - tau(mu, L, alpha, P) V(x),
        for all L-smooth, mu-strongly convex functions, and all solution x to the gradient flow.

        When P=None, this code optimizes over the family of quadratic Lyapunov function.

        :param mu: strong convexity assumption
        :param L: smoothness assumption
        :param alpha: positive ODE parameter
        :param a: lyapunov parameter (one has to be fixed)
        :param upper: upper bound on the convergence rate
        :param lower: lower bound on the convergence rate
        :param epsilon: precision
        :return:
            - worst-case guarantee tau(L, mu, alpha, P)
            - P
            - dual variables associated with interpolation inequalities

        """
        ## PARAMETERS
        npt = 2  # optimal and initial point
        dimF = 1  # one dimension only
        dimG = 2  # (x(t) // grad f(x(t)))

        ## INITIALIZE with optimal point x_star
        FF, GG, YY = [np.zeros((1, dimF))], [np.zeros((1, dimG))], [np.zeros((1, dimG))]
        # define the point x(t)
        y, f, g = np.zeros((1, dimG)), np.zeros((1, dimF)), np.zeros((1, dimG))
        y[0][0], f[0][0], g[0][1] = 1., 1., 1.
        YY.append(y)
        FF.append(f)
        GG.append(g)

        # Compute the ODE
        X_t = YY[1]  # X(t)
        X_dot = - alpha * GG[1]  # dot(X(t)) = - alpha * grad(f)(X(t))

        ## GET THE INEQUALITIES
        A = []
        b = []
        for i in range(npt):
            for j in range(npt):
                if j != i:
                    if (mu != 0 & L != 0):
                        Aij = np.dot((YY[i] - YY[j]).T, GG[j]) + \
                              1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                                      mu * np.dot((YY[i] - YY[j]).T, YY[i] - YY[j]) -
                                                      2 * mu / L * np.dot((YY[i] - YY[j]).T, GG[i] - GG[j]))
                    if (mu != 0 & L == 0):
                        Aij = np.dot((YY[i] - YY[j]).T, GG[j]) + \
                              1 / 2 * mu * np.dot((YY[i] - YY[j]).T, YY[i] - YY[j])
                    if (mu == 0 & L != 0):
                        Aij = np.dot((YY[i] - YY[j]).T, GG[j]) + \
                              1 / 2 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j])
                    if (mu == 0 & L == 0):
                        Aij = np.dot((YY[i] - YY[j]).T, GG[j])
                    A.append(.5 * (Aij + Aij.T))
                    b.append(FF[j] - FF[i])

        if P == None :
            # If there is no value for P, we optimize over the class of quadratic Lyapunov functions
            l_ = None
            P_ = P
            while upper - lower >= epsilon:
                tau = (lower + upper) / 2
                # Variables in CVXPY
                P = cp.Variable(1)
                l = cp.Variable(npt * (npt - 1))  # dual values : interpolation inequalities
                # CONSTRAINTS
                ### Positivity of the interpolation inequalities
                constraints = [l <= 0.]
                ### Constraints for positivity of the Lyapunov
                constraints = constraints + [P >= 0.]  # P is positive
                constraints = constraints + [sum([l[i] * A[i] for i in range(len(A))])
                                             + P * (X_dot.T @ X_t + X_t.T @ X_dot)
                                             + a * (X_dot.T @ GG[1] + GG[1].T @ X_dot) / 2
                                             + tau * P * X_t.T @ X_t << 0.]  # derivative of f(x(t))
                constraints = constraints + [sum([l[i] * b[i][0] for i in range(len(b))])
                                             + tau * (a * FF[1][0]) == 0.]
                # OPTIMIZE
                prob = cp.Problem(cp.Minimize(0.), constraints)
                prob.solve(solver=cp.SCS)
                if prob.status == 'optimal':
                    lower = tau
                    P_ = P.value
                    l_ = l.value
                else:
                    upper = tau

        else:
            P_ = P
            l_ = None

            while upper - lower >= epsilon:
                tau = (lower + upper) / 2
                # Variables in CVXPY
                l = cp.Variable(npt * (npt - 1)) # dual values : interpolation inequalities
                # CONSTRAINTS
                ### Positivity of the interpolation inequalities
                constraints = [l <= 0]
                constraints = constraints + [sum([l[i] * A[i] for i in range(len(A))])
                                             + P * (X_dot.T @ X_t + X_t.T @ X_dot)
                                             + a * (X_dot.T @ GG[1] + GG[1].T @ X_dot) / 2
                                             + tau * P * X_t.T @ X_t << 0.]  # derivative of f(x(t))
                constraints = constraints + [sum([l[i] * b[i][0] for i in range(len(b))])
                                             + tau * (a * FF[1][0]) == 0.]
                # OPTIMIZE
                prob = cp.Problem(cp.Minimize(0.), constraints)
                prob.solve(solver=cp.SCS)
                if prob.status == 'optimal':
                    lower = tau
                    l_ = l.value
                else:
                    upper = tau

        return tau, P_, l_


if __name__ == '__main__':
    # ODE parameter
    alpha = 1
    # Function class
    L = 0 # smoothness parameter
    mu = 0.0001 # strong-convexity parameter
    # Lyapunov parameters
    P = 0 # Lyapunov parameter
    a = 1

    tau, _, l = compute_convergence_guarantee(mu=mu,
                                              L=L,
                                              alpha=alpha,
                                              a=a,
                                              P=P)
    print('Convergence guarantee for a given P > 0 ', tau, 2 * mu)

    tau, P, _,  = compute_convergence_guarantee(mu=mu,
                                              L=L,
                                              alpha=alpha,
                                              a=a,
                                             P=None)
    print(l)
    print('Convergence guarantee while optimizing over P ', tau, 2 * mu)
    print(P)