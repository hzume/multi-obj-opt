import cvxpy as cp
import numpy as np
from problems import MultiObjProblem, JOS1, KW2
from matplotlib import pyplot as plt
import jax.numpy as jnp
from jax import Array

class ProjectedGradDescent:
    def __init__(
            self, 
            prob: MultiObjProblem,
            nu: float = 0.1,
            sigma: float = 0.5,
            beta: float = 1,
            ) -> None:
        self.nu = nu
        self.sigma = sigma
        self.n = prob.n
        self.m = prob.m
        self.F = prob.__call__
        self.J = prob.J
        self.L = prob.L
        self.U = prob.U
        self.beta = beta
    
    def step(self, x_old: Array) -> Array:
        d = self.descent_direction(x_old)
        step_size = self.nu
        while True: 
            x_new = x_old + step_size * d
            if jnp.all(self.F(x_new) <= self.F(x_old) + self.sigma * step_size * self.J(x_old) @ d):
                break
            step_size *= self.nu
        return x_new

    def descent_direction(self, x: Array) -> Array:
        d = cp.Variable(self.n)
        sub_obj = cp.Minimize(self.beta * cp.max(self.J(x)@d) + cp.sum_squares(d)/2)
        sub_constraint = [self.L - x <= d, d <= self.U - x]
        sub_prob = cp.Problem(sub_obj, sub_constraint)
        sub_prob.solve()
        d = d.value
        self.d = d
        self.theta = sub_prob.value
        assert d.shape == (self.n,)
        return -d
    
    def solve(self, x_ini: Array, epsilon: float=1e-5, verbose=True) -> Array:
        x_old = x_ini
        i = 0
        np.set_printoptions(precision=6, floatmode="fixed", suppress=True)
        while True:
            x_new = self.step(x_old)
            if jnp.abs(self.theta) < epsilon:
                break
            if (i % 100 == 0) & verbose:
                print(f"\rtheta: {self.theta:.6f} values: {self.F(x_new)} descent: {self.d}", end="")
            x_old = x_new
            i += 1
        if verbose: print("\n")
        print(f"Finished.")
        print(f"steps: {i} theta: {self.theta:.6f} values: {self.F(x_new)} descent: {self.d}")
        return x_new, self.F(x_new)
    
if __name__=="__main__":
    n = 2
    m = 2
    L = jnp.ones(n)*(-2)
    U = jnp.ones(n)*2
    problem = JOS1(n, m, L, U)
    solver = ProjectedGradDescent(problem)
    # solver.solve(np.array([-1, 2, -2, 1, 1]*2), epsilon=1e-6)
    x_1_arr = jnp.linspace(-2, 2, 100)
    x_2_arr = jnp.linspace(-2, 2, 100)
    F = jnp.array([problem(jnp.array([x_1, x_2]), True) for x_1 in x_1_arr for x_2 in x_2_arr])
    plt.plot(F[:,0], F[:,1])
    optimal_values = []
    for _ in range(10):
        x = jnp.array((np.random.rand(n)-0.5)*4)
        optimizer, optimal_value = solver.solve(x, verbose=True, epsilon=1e-3)
        optimal_values.append(optimal_value)
    optimal_values = jnp.array(optimal_values)
    plt.plot(optimal_values[:,0], optimal_values[:,1], ".")
    plt.savefig("b.png")