# PEP_ODEs

This project aims at transfering discrete optimization techniques to ordinary differential equations (ODEs), called Performance 
Estimation Problems (PEPs). First-oder methods are often approached via their continuous-time models. In both continuous and 
discrete time, worst-case convergence guarantees are often derived using Lyapunov functions. The following code provides a numerical 
tool for analyzing gradient flows originating from strongly convex functions, via a specific family of quadratic Lyapunov functions. 

Given such a Lyapunov function, we formulate the search for convergence guarantee in the worst-case as a small-sized semidefinite 
program. Our formulation also allows to find the fastest convergence rate that can be achieved using quadratic Lyapunov 
functions. More details are given in the references.

We analyze the following gradient flows: 
- first-order gradient flow (for gradient descent),
- second-order gradient flow, or Polyak's damped oscillator equation (for gradient-based accelerated methods).

This code allows to graw figures from the paper : **A systematic approach to Lyapunov analyses of continuous-time models 
in convex optimization** (to come) - C. Moucer, A. Taylor, F. Bach


## Getting Started


### Prerequisites
This codes relies on the following python modules, with versions provided in requirements.txt:

- numpy
```
pip install numpy
```
- cvxpy
```
pip install cvxpy
```
- matplotlib
```
pip install matplotlib
```

We recommend to use the following solver:
- solver MOSEK 
```
pip install mosek
```
Otherwise:
- solver SCS (already installed in cvxpy)

## Compute convergence guarantees
Our code is divided into two parts : first-order gradient flows and second-order gradient flows. Each file contains
a main.py that allows to retrieve figures from the paper.

Our code allows to compute : 
- worst-case convergence guarantees for a given quadratic Lyapunov function,
- worst-case convergence guarantees while optimizing over a class of quadratic Lyapunov functions,
- a strongly-convex function that matches a worst-case guarantee (for first-order flows).


## Authors
* **Céline MOUCER** 
* **Adrien TAYLOR**
* **Francis BACH** 

## References

* [PESTO - Matlab](https://github.com/PerformanceEstimation/Performance-Estimation-Toolbox) - A. Taylor
* [PEPit - Python](https://github.com/PerformanceEstimation/PEPit) - B. Goujaud, C. Moucer, J. Hendricks, F. Glineur, A. Taylor, A. Dieuleveut
* [Performance Estimation Problems](https://arxiv.org/abs/1206.3209) - Y. Drori, M. Teboulle
* [Smooth Strongly Convex Interpolation and Exact Worst-case Performance of First-order Methods](https://arxiv.org/abs/1502.05666) - A. Taylor, J. Hendrickx, F. Glineur 
* A systematic approach to Lyapunov analyses of continuous-time models in convex optimization (to come) - C. Moucer, A. Taylor, F. Bach
