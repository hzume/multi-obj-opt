import cvxpy as cp
import numpy as np
from problem import MultiObjProblem, JOS1
from matplotlib import pyplot as plt

class SteepestGradDescent:
    def __init__(
            self, 
            prob: MultiObjProblem, 
            nu: float = 0.1,
            sigma: float = 0.5 
            ) -> None:
        self.nu = nu
        self.sigma = sigma
        self.n = prob.n
        self.m = prob.m
        self.F = prob.__call__
        self.J = prob.J
    
    def step(self, x_old: np.ndarray) -> np.ndarray:
        d = self.descent_direction(x_old)
        step_size = self.nu
        while True: 
            x_new = x_old + step_size * d
            if np.all(self.F(x_new) <= self.F(x_old) + self.sigma * step_size * self.J(x_old) @ d):
                break
            step_size *= self.nu
        return x_new

    def descent_direction(self, x: np.ndarray) -> np.ndarray:
        w = cp.Variable(self.m)
        sub_obj = cp.Minimize(cp.sum_squares(self.J(x).T @ w))
        sub_constraint = [cp.sum(w) == 1, 0 <= w, w <= 1]
        sub_prob = cp.Problem(sub_obj, sub_constraint)
        sub_prob.solve()
        d = w.value
        self.d = d
        self.theta = sub_prob.value
        assert d.shape == (self.m,)
        return - self.J(x).T @ d
    
    def solve(self, x_ini: np.ndarray, epsilon: float=1e-2, verbose=True) -> np.ndarray:
        x_old = x_ini
        i = 0
        np.set_printoptions(precision=6, floatmode="fixed", suppress=True)
        while True:
            x_new = self.step(x_old)
            if self.theta < epsilon:
                break
            if (i % 100 == 0) & verbose:
                print(f"\rtheta: {self.theta:.6f} values: {self.F(x_new)} descent: {self.d}", end="")
            x_old = x_new
            i += 1
        print(f"\nFinished.")
        print(f"steps: {i} theta: {self.theta:.6f} values: {self.F(x_new)} descent: {self.d}")
        return x_new, self.F(x_new)
    
if __name__=="__main__":
    problem = JOS1(1, 2)
    solver = SteepestGradDescent(problem)
    F = np.array([problem(np.array([1e-3*i])) for i in range(5000)])
    plt.plot(F[:,0], F[:,1])
    optimal_values = []
    for _ in range(30):
        optimizer, optimal_value = solver.solve((np.random.rand(1)-0.5)*5, epsilon=1e-8, verbose=False)
        optimal_values.append(optimal_value)
    optimal_values = np.array(optimal_values)
    plt.plot(optimal_values[:,0], optimal_values[:,1], ".")
    plt.savefig("a.png")