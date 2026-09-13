"""Standalone copula structure and caller-controlled fitting. Template keys identify shared marginal transforms; supplied paired observations determine admitted correlations. These utilities neither construct canonical evidence likelihoods nor fit parameters during inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef, Family
from ocbf.model.copula.bridge import MAX_ABS_CORRELATION, bridge_correlation, kendall_tau
from ocbf.model.copula.marginals import GaussianMarginal, MarginalTransform, fit_marginal
from ocbf.schema import AttributeKind, Schema

EVENT_TIME_KEY = "event_time"
"""Shared template key for caller-supplied timestamp marginal transforms."""


def marginal_key(ref: AssertionRef) -> str:
    """Template key of a continuous assertion.

    Attributes key on their name, which OCEL 2.0 guarantees is unique across types (the
    schema enforces it), so the name alone determines the transform.

    Raises:
        ValueError: for a ref whose family has no place in the Gaussian block.
    """
    match ref.family:
        case Family.EVENT_TIME:
            return EVENT_TIME_KEY
        case Family.EVENT_ATTR | Family.OBJECT_ATTR:
            assert ref.attribute is not None  # guaranteed by AssertionRef validation
            return ref.attribute
        case _:
            raise ValueError(f"{ref.family.value} is not a continuous family")


@dataclass(frozen=True, slots=True)
class CopulaSpec:
    """Marginal transforms and latent correlations, keyed by template.

    Correlations are stored on the *unordered* template pair, so a lookup is symmetric and
    a pair declared once cannot disagree with itself. An absent pair is independent.
    """

    marginals: Mapping[str, MarginalTransform]
    correlations: Mapping[frozenset[str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for pair, rho in self.correlations.items():
            if len(pair) > 2:
                raise ValueError(f"a correlation names at most two templates, got {sorted(pair)}")
            if not -1.0 < rho < 1.0:
                raise ValueError(f"correlation for {sorted(pair)} must be in (-1, 1), got {rho}")
            missing = pair - set(self.marginals)
            if missing:
                raise ValueError(f"correlation names templates without marginals: {sorted(missing)}")

    def has(self, ref: AssertionRef) -> bool:
        """Whether this ref's template has a fitted marginal."""
        try:
            return marginal_key(ref) in self.marginals
        except ValueError:
            return False

    def marginal(self, ref: AssertionRef) -> MarginalTransform:
        """Transform for one continuous assertion.

        Raises:
            KeyError: if the template was never fitted. Deliberately not a silent standard
                normal -- reporting a timestamp on an unfitted marginal would produce
                plausible numbers in the wrong units.
        """
        key = marginal_key(ref)
        try:
            return self.marginals[key]
        except KeyError:
            raise KeyError(f"no marginal fitted for template {key!r}") from None

    def correlation(self, left: str, right: str) -> float:
        """Latent correlation between two templates; 0.0 when undeclared."""
        if left == right:
            return 0.0
        return float(self.correlations.get(frozenset((left, right)), 0.0))

    def summary(self) -> dict[str, object]:
        """Which templates are modelled, and how strongly they are coupled."""
        rhos = np.fromiter(self.correlations.values(), dtype=np.float64, count=len(self.correlations))
        return {
            "templates": len(self.marginals),
            "kinds": sorted({m.kind.value for m in self.marginals.values()}),
            "correlated_pairs": len(self.correlations),
            "max_abs_correlation": round(float(np.abs(rhos).max()), 4) if rhos.size else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"CopulaSpec(templates={len(self.marginals)}, pairs={len(self.correlations)})"
        )


def coupling_precision(rho: float) -> np.ndarray:
    """The ``2x2`` precision block one correlated pair contributes.

    Each latent coordinate already carries its own standard-normal prior, so a pair must
    contribute only the *difference* between the bivariate precision and those two unit
    marginals:

    ```text
    Theta_pair = 1/(1 - rho^2) * [[1, -rho], [-rho, 1]]      (the bivariate precision)
    coupling   = Theta_pair - I
    ```

    Splitting it this way is what keeps the model coherent when a coordinate takes part in
    several pairs: the priors are counted once, in the graph, and each edge adds only its
    own coupling. Summing full bivariate precisions instead would count every marginal once
    per edge and shrink the block toward zero for no reason.
    """
    r = float(np.clip(rho, -MAX_ABS_CORRELATION, MAX_ABS_CORRELATION))
    denom = 1.0 - r * r
    off = -r / denom
    diag = r * r / denom
    return np.array([[diag, off], [off, diag]], dtype=np.float64)


def fit_copula(
    observations: Mapping[str, Sequence[float] | np.ndarray],
    *,
    schema: Schema | None = None,
    pairs: Iterable[tuple[str, str, Sequence[float], Sequence[float]]] = (),
    kinds: Mapping[str, AttributeKind] | None = None,
    seed: int = 0,
) -> CopulaSpec:
    """Fit standalone marginal transforms and latent correlations from supplied observations.

    ``observations`` pools values by template; co-observed pairs inform correlations. Timestamp coordinates use the explicitly supplied Gaussian time assumptions. Fitting is caller-controlled and does not run inside canonical inference."""
    marginals: dict[str, MarginalTransform] = {}
    for key, values in observations.items():
        if key == EVENT_TIME_KEY:
            marginals[key] = GaussianMarginal.fit(values)
            continue
        spec = None
        if schema is not None:
            try:
                spec = schema.attribute_owner(key)[1]
            except KeyError:
                spec = None
        kind = spec.kind if spec is not None else (kinds or {}).get(key, AttributeKind.CONTINUOUS)
        marginals[key] = fit_marginal(values, kind=kind, spec=spec)

    correlations: dict[frozenset[str], float] = {}
    for left, right, x, y in pairs:
        if left == right or left not in marginals or right not in marginals:
            continue
        tau = kendall_tau(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        rho = bridge_correlation(tau, marginals[left], marginals[right], seed=seed)
        if abs(rho) > 1e-6:
            correlations[frozenset((left, right))] = rho

    return CopulaSpec(marginals, correlations)
