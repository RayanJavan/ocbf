"""The assertion algebra: how the latent world is addressed.

Design doc section 1.1 lifts every component of an OCEL 2.0 log to a random variable.
This package supplies the two faces of that lift:

* [`AssertionRef`][ocbf.assertions.refs.AssertionRef] -- the human-facing, typed, hashable *address* of one latent
  quantity. Stable, printable, and what sources speak in.
* [`VariableRegistry`][ocbf.assertions.registry.VariableRegistry] -- the machine-facing side: contiguous integer ids grouped by
  family, with parallel numpy arrays. At 1e7 variables the model cannot afford to touch a
  Python object per message, so every hot path works on the arrays.
"""

from ocbf.assertions.refs import AssertionRef, Family
from ocbf.assertions.registry import VariableRegistry, VarKind

__all__ = ["AssertionRef", "Family", "VarKind", "VariableRegistry"]
