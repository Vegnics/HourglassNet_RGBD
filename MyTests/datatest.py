import sys
from typing import Any
import numpy as np
class Matros():
    def __init__(self,x) -> None:
        self.x = x
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.x**2

a = np.array([1,2,3,4,5,6])
b = np.array([False,True,True,True,True,False],dtype=bool)
print(a[b])
