from typing import Dict
import cvxpy as cp
import numpy as np
from problems import MultiObjProblem, JOS1, JOS2, KW2, KW3, ConstrainedMOP
from matplotlib import pyplot as plt
import jax.numpy as jnp
from jax import Array

class PrimalDual:
    def __init__(
            self, 
            prob: ConstrainedMOP,
            nu: float = 0.1,
            sigma: float = 0.5,
            # beta: float = 1,
            ) -> None:
        self.nu = nu
        self.sigma = sigma
        self.n = prob.n
        self.m = prob.m
        self.k = prob.k
        self.F = prob.F
        self.L = prob.__call__
        self.G = prob.G
        self.J = prob.J
        self.beta = None
    
    def step(self, params_old: Dict[str, Array]) -> Array:
        d = self.descent_direction(params_old)
        step_size = self.nu
        while True: 
            x_new = params_old["x"] + step_size * d[:self.n]
            lambda_new = params_old["lambda"] + step_size * d[self.n:]
            params_new = {"x": x_new, "lambda": lambda_new}
            if jnp.all(self.L(params_new) <= self.L(params_old) + self.sigma * step_size * self.J(params_old) @ d):
                break
            step_size *= self.nu
        return params_new
    
    def descent_direction(self, params: Dict[str, Array]) -> Array:
        d = cp.Variable(self.n + self.k)
        sub_obj = cp.Minimize(self.beta * cp.max(self.J(params)@d) + cp.sum_squares(d)/2)
        sub_constraint = [
            params["lambda"] + d[self.n:] >= 0,
            ]
        sub_prob = cp.Problem(sub_obj, sub_constraint)
        sub_prob.solve()
        d = d.value
        self.d = d
        self.theta_new = sub_prob.value
        assert d.shape == (self.n + self.k,)
        return d
    
    def solve(self, params_ini: Dict[str, Array], epsilon: float=1e-5, verbose=True, beta=0.5) -> Array:
        self.beta = beta
        params_old = params_ini
        self.theta_old = -np.inf
        i = 0
        np.set_printoptions(precision=6, floatmode="fixed", suppress=True)
        while True:
            self.beta = 1 / (i + 1)
            params_new = self.step(params_old)
            if jnp.abs(self.theta_new) < epsilon:
                break
            if (i % 10 == 0) & verbose:
                print(f"\rtheta: {self.theta_new:.6f} value: {self.F(params_new['x'])} descent: {self.d}" , end="")
            params_old = params_new
            self.theta_old = self.theta_new
            i += 1
        if verbose: print("\n")
        print(f"Finished.")
        print(f"steps: {i} theta: {self.theta_new:.6f} values: {self.F(params_new['x'])} descent: {self.d}")
        return params_new, self.L(params_new)

if __name__=="__main__":
    n = 2
    m = 2
    d = 2
    problem = JOS2(n, m, d)
    solver = PrimalDual(problem)
    x_1_arr = jnp.linspace(-2, 2, 100)
    x_2_arr = jnp.linspace(-2, 2, 100)
    F = jnp.array([problem.F(jnp.array([x_1, x_2])) for x_1 in x_1_arr for x_2 in x_2_arr])
    plt.plot(F[:,0], F[:,1])
    optimal_values = []
    for _ in range(10):
        x = jnp.array((np.random.rand(n)-0.5)*4)
        Lambda = jnp.array((np.random.rand(d))*0)
        params = {"x":x, "lambda":Lambda}
        optimizer, optimal_value = solver.solve(params, verbose=True, epsilon=1e-5)
        optimal_values.append(problem.F(optimizer["x"]))
    optimal_values = jnp.array(optimal_values)
    plt.plot(optimal_values[:,0], optimal_values[:,1], ".")
    plt.savefig("d.png")