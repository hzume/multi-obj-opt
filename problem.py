from typing import Any
from nptyping import NDArray, Shape
import numpy as np
import cvxpy
from abc import ABC, abstractmethod

class MultiObjProblem(ABC):
    def __init__(self, n:int, m:int) -> None:
        self.n = n
        self.m = m

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def J(self, x: np.ndarray) -> np.ndarray:
        pass

class JOS1(MultiObjProblem):
    def __call__(self, x: np.ndarray) -> None:
        assert x.shape == (self.n,)
        F = np.array([np.sum((x - 2*i)**2) / self.n for i in range(self.m)])
        assert F.shape == (self.m,)
        return F
    
    def J(self, x):
        assert x.shape == (self.n,)
        grad_F = np.array([(x - 2*i) * 2 / self.n for i in range(self.m)])
        assert grad_F.shape == (self.m, self.n)
        return grad_F

if __name__=="__main__":
    F = JOS1(2, 3)
    print(F.J(np.array([2, 2])))