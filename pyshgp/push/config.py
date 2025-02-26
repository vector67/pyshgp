"""Push program execution requires configuration to ensure all Push programs are safe.

Synthesizing Push requires generating random code. When executed, this random code can contain a number of
bad behaviors such as: exponential memory utilization, infinite loops, and impractically long runtime. This
modules defines important configuration objects that are used to constrain push program execution.

A block of Push code can produce different behavior under different ``PushConfig`` settings. This implies that
the ``PushConfig`` used to produce a solution program must be a part of the definition of a push ``Program``.

"""
from collections.abc import Sequence
from typing import Union

from pyrsistent import PRecord, field


class PushConfig:
    """A configuration for a Push program.

    Attributes
    ----------
    step_limit : int
        Max number of atoms to process before terminating the execution of a
        given program. Default is 500
    runtime_limit : int
        Max number of seconds to run a push program before forcing
        termination. Default is 10.
    growth_cap : int
        Max number of elements that can be added to a PushState at any given
        step of program execution. If exceeded, program terminates. Default is 500.
    collection_size_cap : int, optional
        Max size of any collection (code blocks, vectors, strings, etc). Default is 1000.
    numeric_magnitude_limit : float, optional
        Max magnitude of numbers. Default is 1 million.

    """

    step_limit: int = 500
    runtime_limit: int = 10
    growth_cap: int = 500
    collection_size_cap: int = 1000
    numeric_magnitude_limit: float = 1e12

    def __init__(self, step_limit=500, runtime_limit=10, growth_cap=500, collection_size_cap=1000,
                 numeric_magnitude_limit=1e12):
        self.step_limit = step_limit
        self.runtime_limit = runtime_limit
        self.growth_cap = growth_cap
        self.collection_size_cap = collection_size_cap
        self.numeric_magnitude_limit = numeric_magnitude_limit


def constrain_collection(config: PushConfig, coll: Sequence) -> Sequence:
    """Constrains the collection to a size that is safe for Push program execution."""
    if len(coll) > config.collection_size_cap:
        return coll[:config.collection_size_cap]
    return coll


def constrain_number(config: PushConfig, n: Union[int, float]) -> Union[int, float]:
    """Constrains the number to a magnitude that is safe for Push program execution."""
    if abs(n) > config.numeric_magnitude_limit:
        sign = -1 if n < 0 else 1
        return config.numeric_magnitude_limit * sign
    return n
