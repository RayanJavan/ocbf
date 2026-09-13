"""Reuse elastocel's structure derivation; no copy of its catalogue inference logic."""

from ocbf.errors import CapabilityError, ValidationError
from ocbf.universe.context import SemanticContext


def semantic_context(catalog, log, *, config, provenance):
    """Accept existing elastocel ports, with explicit temporal/candidate configuration."""
    if not provenance:
        raise ValidationError("historical catalogue and binding provenance are required")
    try:
        from elastocel.structure import derive_structure
    except ImportError as exc:
        raise CapabilityError(
            "batch integration requires elastocel; generic OCBF does not"
        ) from exc
    result = derive_structure(catalog, log, config=config)
    if result.universe is None:
        raise ValidationError("structure bridge did not produce candidate support")
    return SemanticContext.from_universe(
        result.universe,
        constraints=result.constraints,
        provenance={
            **provenance,
            "structure_bridge": "elastocel.structure.derive_structure",
            "candidate_report": str(result.report),
        },
    )
