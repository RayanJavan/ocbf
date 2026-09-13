"""Immutable manual parameters for association-mode and Gaussian timestamp channels."""

import math
from dataclasses import dataclass

from ocbf._values import freeze
from ocbf.errors import ValidationError


@dataclass(frozen=True)
class AssociationParameters:
    """Rows are candidate truths; columns are reports, separately for each shared mode."""

    modes: tuple
    mode_prior: tuple[float, ...]
    matrices: tuple
    mode_key: str
    origin: str

    def __post_init__(self):
        import numpy as np

        for name in ("modes", "mode_prior", "matrices"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        p = np.asarray(self.mode_prior, dtype=float)
        a = np.asarray(self.matrices, dtype=float)
        if (
            not self.origin
            or not self.mode_key
            or not self.modes
            or len(set(self.modes)) != len(self.modes)
            or p.shape != (len(self.modes),)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
            or a.ndim != 3
            or a.shape[0] != len(self.modes)
            or not np.isfinite(a).all()
            or (a < 0).any()
            or not np.allclose(a.sum(axis=2), 1, atol=1e-12, rtol=0)
        ):
            raise ValidationError("invalid mode prior or association confusion matrices")


@dataclass(frozen=True)
class TimestampParameters:
    sd: float
    bias: float
    unit: str
    origin: str
    clock_key: str | None = None
    clock_sd: float | None = None
    inactive_log_likelihood: float | None = None

    def __post_init__(self):
        if (
            not self.origin
            or not self.unit
            or not math.isfinite(self.sd)
            or self.sd <= 0
            or not math.isfinite(self.bias)
            or (self.clock_key is None) != (self.clock_sd is None)
            or self.clock_sd is not None
            and (not math.isfinite(self.clock_sd) or self.clock_sd <= 0)
            or self.inactive_log_likelihood is not None
            and not math.isfinite(self.inactive_log_likelihood)
        ):
            raise ValidationError("timestamp error needs explicit proper parameters and units")
