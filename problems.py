from typing import Any, Dict
from nptyping import NDArray, Shape
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, Array
import cvxpy
from abc import ABC, abstractmethod

class MultiObjProblem(ABC):
    def __init__(self, n:int, m:int, L: Array = None, U: Array = None) -> None:
        self.n = n
        self.m = m
        self.L = L
        self.U = U
        self.grad_F = jacfwd(self.F)

    @abstractmethod
    def F(self, x: Array) -> Array:
        pass

    def __call__(self, x: Array, ignore_constraint: bool=False) -> Array:
        if jnp.all(self.L!=None) and not ignore_constraint:
            assert jnp.all(self.L <= x)
        if jnp.all(self.U!=None) and not ignore_constraint:
            assert jnp.all(x <= self.U)

        assert x.shape == (self.n,)
        F = self.F(x)
        assert F.shape == (self.m,)
        return F

    def J(self, x: Array, ignore_constraint: bool=False) -> Array:
        if jnp.all(self.L!=None) and not ignore_constraint:
            assert jnp.all(self.L <= x)
        if jnp.all(self.U!=None) and not ignore_constraint:
            assert jnp.all(x <= self.U)
        
        assert x.shape == (self.n,)
        grad_F = self.grad_F(x)
        assert grad_F.shape == (self.m, self.n)
        return grad_F

class ConstrainedMOP(MultiObjProblem):
    def __init__(self, n: int, m: int, k: int) -> None:
        self.n = n
        self.m = m
        self.k = k
        self.grad_F = jacfwd(self.L)

    @abstractmethod
    def G(self, x: Array):
        pass
    
    def L(self, params: Dict[str, Array]) -> Array:
        if self.k == 1:
            return self.F(params["x"]) + params["lambda"].T * self.G(params["x"])
        else:
            return self.F(params["x"]) + params["lambda"].T @ self.G(params["x"])

    def __call__(self, params: Dict[str, Array], ignore_constraint: bool = False) -> Array:
        assert params["x"].shape == (self.n,)
        assert params["lambda"].shape == (self.k,)
        # assert jnp.all(self.G(params["x"]) <= 1e-6)
        L = self.L(params)
        assert L.shape == (self.m,)
        return L

    def J(self, params: Dict[str, Array]) -> Array:
        assert params["x"].shape == (self.n,)
        grad_L_dict = self.grad_F(params)
        grad_L_x = grad_L_dict["x"]
        grad_L_lambda = grad_L_dict["lambda"]
        assert grad_L_x.shape == (self.m, self.n)
        assert grad_L_lambda.shape == (self.m, self.k)
        grad_L = jnp.concatenate([grad_L_x, grad_L_lambda], axis=1)
        assert grad_L.shape == (self.m, self.n + self.k)
        return grad_L
    
class JOS1(MultiObjProblem):
    def F(self, x: Array) -> Array:
        F = jnp.array([jnp.sum((x - 2*i)**2) / self.n for i in range(self.m)])
        return F

class JOS2(ConstrainedMOP):
    def F(self, x: Array) -> Array:
        F = jnp.array([jnp.sum((x - 2*i)**2) / self.n for i in range(self.m)])
        return F
    
    def G(self, x: Array) -> Array:
        G = jnp.array([jnp.abs(x[i]) - 2 for i in range(self.k)])
        return G
    
class KW2(MultiObjProblem):
    def __init__(self, n: int=2, m: int=2, L: Array = None, U: Array = None) -> None:
        super().__init__(n, m, L, U)
        assert (n==2) & (m==2)
        
    def F(self, x: Array) -> Array:
        x_1, x_2 = x[0], x[1]

        F_1 = -3 * (1 - x_1)**2 * jnp.exp(-x_1**2 - (x_2+1)**2)\
            + 10 * (x_1/5 - x_1**3 - x_2**5) * jnp.exp(-x_1**2 - x_2**2)\
            + 3 * jnp.exp(-(x_1+2)**2 - x_2**2) - 0.5 * (2*x_1 + x_2)

        F_2 = -3 * (1 + x_2)**2 * jnp.exp(-x_2**2 - (1 - x_1)**2)\
            + 10 * (-x_2/5 + x_2**3 + x_1**5) * jnp.exp(-x_1**2 - x_2**2)\
            + 3 * jnp.exp(-(2-x_2)**2 - x_1**2)

        F = jnp.array([F_1, F_2])
        return F

class KW3(ConstrainedMOP):
    def __init__(self, n: int=2, m: int=2, k: int=2) -> None:
        super().__init__(n, m, k)
        assert (n==2) & (m==2)
    
    def G(self, x: Array) -> Array:
        G = jnp.array([jnp.abs(x[i]) - 3 for i in range(self.k)])
        return G
        
    def F(self, x: Array) -> Array:
        x_1, x_2 = x[0], x[1]

        F_1 = -3 * (1 - x_1)**2 * jnp.exp(-x_1**2 - (x_2+1)**2)\
            + 10 * (x_1/5 - x_1**3 - x_2**5) * jnp.exp(-x_1**2 - x_2**2)\
            + 3 * jnp.exp(-(x_1+2)**2 - x_2**2) - 0.5 * (2*x_1 + x_2)

        F_2 = -3 * (1 + x_2)**2 * jnp.exp(-x_2**2 - (1 - x_1)**2)\
            + 10 * (-x_2/5 + x_2**3 + x_1**5) * jnp.exp(-x_1**2 - x_2**2)\
            + 3 * jnp.exp(-(2-x_2)**2 - x_1**2)

        F = jnp.array([F_1, F_2])
        return F

if __name__=="__main__":
    prob = JOS2(3, 3, 2)
    params = {"x":jnp.array([-1.0, 1.0, 3.0]), "lambda":jnp.array([1.0, 1.0])}
    grad_x, grad_lambda, grad_L = prob.J(params)
    print("x: ")
    print(grad_x, "\n")
    print("lambda: ")
    print(grad_lambda, "\n")
    print(grad_L)
    