import sys
from typing import Any
class Matros():
    def __init__(self,x) -> None:
        self.x = x
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.x**2

m = Matros(10)
print(m(),m.x)
