import numpy as np
import cvxpy as cp

def compute_convergence_guarantee(mu, alpha=1, a=1., c = None,  upper=2., lower=0., epsilon=10 ** -5, verbose=False):
    """
    Consider the gradient flow
            dx/dt = - alpha * grad f(x(t)),
    where alpha is a positive parameter, f a mu-strongly convex function, and x_* a stationary point.

    We consider quadratic Lyapunov functions
            V(x(t)) = a(f(x(t)) - f_*) + c ||x(t) - x_*||^2,
    where a,c are positive parameters.

    This code compute the worst-case guarantee tau(mu, alpha, c) that verifies
            d/dt V(x) <= - tau(mu, alpha, c) V(x),
    for all mu-strongly convex functions, and all solution x to the gradient flow.

    When P=None, this code optimizes over the family of quadratic Lyapunov function.

    :param mu: strong convexity assumption
    :param alpha: positive ODE parameter
    :param a: lyapunov parameter (one has to be fixed)
    :param upper: upper bound on the convergence rate
    :param lower: lower bound on the convergence rate
    :param epsilon: precision
    :return:
        - worst-case guarantee tau(mu, alpha, c)
        - c
        - dual variables associated with interpolation inequalities

    """

    if c == None or a == None:
        # If there is no value for P, we optimize over the class of quadratic Lyapunov functions
        l_ = None
        c_ = c
        a_ = a
        while upper - lower >= epsilon:
            tau = (lower + upper) / 2
            # Variables in CVXPY
            S = cp.Variable((2, 2), symmetric=True)
            c = cp.Variable(1)
            a = cp.Variable(1)
            l = cp.Variable(2)  # dual values : interpolation inequalities
            # CONSTRAINTS
            ### Positivity of the interpolation inequalities
            constraints = [l >= 0.]
            ### Positivity and sclaing of the Lyapunov
            constraints = constraints + [a >= 0.]
            constraints = constraints + [c >= 0.]
            constraints = constraints + [a + c == 1.]
            ### LMI constraint
            constraints = constraints + [ S << 0.]
            constraints = constraints + [S[0][0] == tau * c - mu * (l[0] + l[1])/2]
            constraints = constraints + [S[1][0] == -c * alpha + l[0]/2]
            constraints = constraints + [S[1][1] == -a * alpha]
            constraints = constraints + [tau * a == l[0] - l[1]]
            # OPTIMIZE
            prob = cp.Problem(cp.Minimize(0.), constraints)
            try:
                prob.solve(solver=cp.MOSEK)  # to improve the result: should require higher mosek precision.
                status = prob.status
            except Exception as e:
                status = 'wrong'  # if MOSEK bugs, problem is declared infeasible.
            if status == 'optimal':
                lower = tau
                c_ = c.value
                l_ = l.value
                if verbose:
                    print('feasible point: interval is now {:.6}'.format(lower), ' ,{:.6}'.format(upper))
            else:
                upper = tau
                if verbose:
                    print('infeasible point: interval is now {:.6}'.format(lower), ' ,{:.6}'.format(upper))

    else:
        # If there is no value for P, we optimize over the class of quadratic Lyapunov functions
        l_ = None
        a_ = a
        c_ = c
        while upper - lower >= epsilon:
            tau = (lower + upper) / 2
            # Variables in CVXPY
            S = cp.Variable((2, 2), symmetric=True)
            l = cp.Variable(2)  # dual values : interpolation inequalities
            # CONSTRAINTS
            ### Positivity of the interpolation inequalities
            constraints = [l >= 0.]
            ### Positivity and sclaing of the Lyapunov
            ### LMI constraint
            constraints = constraints + [S << 0.]
            constraints = constraints + [S[0][0] == tau * c - mu * (l[0] + l[1]) / 2]
            constraints = constraints + [S[1][0] == -c * alpha + l[0] / 2]
            constraints = constraints + [S[1][1] == -a * alpha]
            constraints = constraints + [tau * a == l[0] - l[1]]
            # OPTIMIZE
            prob = cp.Problem(cp.Minimize(0.), constraints)
            try:
                prob.solve(solver=cp.MOSEK)  # to improve the result: should require higher mosek precision.
                status = prob.status
            except Exception as e:
                status = 'wrong'  # if MOSEK bugs, problem is declared infeasible.
            if status == 'optimal':
                lower = tau
                l_ = l.value
                if verbose:
                    print('feasible point: interval is now {:.6}'.format(lower), ' ,{:.6}'.format(upper))
            else:
                upper = tau
                if verbose:
                    print('infeasible point: interval is now {:.6}'.format(lower), ' ,{:.6}'.format(upper))

    return tau, c_, a_, l_

if __name__ == '__main__':
    # ODE parameter
    alpha = 1
    # Function class
    mu = 0.0001  # strong-convexity parameter

    # precision and verbose
    epsilon = 10**-5
    verbose = True

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
    print('Convergence guarantee for given c,a > 0 ', tau, 2 * mu)
    print('Relative error: ', np.abs(2 * mu - tau) / (2 * mu))
    print(' ')

    ## Compute worst-case guarantee while optimizing over Lyapunov functions
    tau, c, a, _,  = compute_convergence_guarantee(mu=mu,
                                                   alpha=alpha,
                                                   epsilon=epsilon,
                                                   verbose=verbose)
    print('Convergence guarantee while optimizing over P ', tau, 2 * mu)
    print('Relative error: ', np.abs(2 * mu - tau) / (2 * mu))
    print(' ')