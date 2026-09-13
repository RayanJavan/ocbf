"""Caller-grounded EP admission and analytical Gaussian reference."""

import numpy as np
import pytest
from ocbf.assertions import AssertionRef as Ref, VariableRegistry
from ocbf.errors import CapabilityError
from ocbf.inference import EPConfig
from ocbf.inference.adapters.approximate import run_ep_adapter
from ocbf.model.gaussian import GaussianGraph
from ocbf.model.gaussian_banks import GaussianEvidenceBank


def test_ep_adapter_matches_gaussian_reference_and_rejects_improper_prior():
    registry = VariableRegistry()
    registry.add_continuous(Ref.event_time("e"))
    registry.freeze()
    graph = GaussianGraph(registry, [0], [GaussianEvidenceBank.from_moments([0], [2.0], [0.5])])
    metadata = {
        "model_id": "external:gaussian-1",
        "evidence_id": "fixture",
        "parameter_id": "fixed",
        "context_id": "fixture",
    }
    result = run_ep_adapter(graph, **metadata, config=EPConfig(damping=0, tol=1e-10))
    assert result.posterior.mean[0] == pytest.approx(4 / 3)
    assert result.posterior.variance[0] == pytest.approx(1 / 3)
    assert result.capabilities == ("gaussian_marginal",)
    assert np.array_equal(graph.prior, [[1, 0]])
    graph.prior[0, 0] = -1
    with pytest.raises(CapabilityError):
        run_ep_adapter(graph, **metadata)
