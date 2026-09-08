"""The continuous layer: copula transforms, Gaussian EP, and the CG coupling.

Structured the same way as `test_exactness.py`, and for the same reason. Expectation
propagation is an approximation, so the tests that matter are the ones that put it beside
something exact -- a closed-form posterior, a numerical integral, a brute-force mixture --
and measure the gap. A test that only checks the engine agrees with itself would pass on a
subtly wrong tilted moment, which is precisely the class of error this layer is exposed to.

Where no oracle exists, the standard is the one design doc section 8.3 sets for the discrete
side: beat the baseline, and be calibrated while doing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from ocbf.assertions import AssertionRef, Family, VariableRegistry
from ocbf.baselines import claim_median, weighted_mean
from ocbf.belief import BeliefStateBuilder
from ocbf.eval import compare_continuous, evaluate_continuous, gaussian_crps, pit_values
from ocbf.inference import BPConfig, EPConfig, run_ep
from ocbf.inference.gabp_ep import cavities, fill_belief_state
from ocbf.model.continuous import (
    ContinuousSpec,
    build_continuous_graph,
    continuous_refs,
    fit_copula_from_claims,
)
from ocbf.model.copula import (
    CopulaSpec,
    EmpiricalMarginal,
    GaussianMarginal,
    OrdinalMarginal,
    TruncatedMarginal,
    bridge_correlation,
    bridge_curve,
    coupling_precision,
    fit_marginal,
    kendall_tau,
    marginal_key,
    truncated_normal_moments,
)
from ocbf.model.gaussian import (
    GaussianGraph,
    gauss_hermite,
    to_moments,
    to_natural,
)
from ocbf.model.gaussian_banks import (
    CGMeanBank,
    CorrelationBank,
    GaussianEvidenceBank,
    IntervalBank,
    ObservationBank,
    PrecedenceBank,
    gaussian_loglik,
    student_t_loglik,
)
from ocbf.pipeline import FusionConfig, fuse
from ocbf.reliability import ReliabilityTable, pairwise_channels
from ocbf.schema import AttributeKind, AttributeSpec, ConstraintClass, Strength
from ocbf.sources import ClaimSet
from ocbf.synth import ProcessConfig, SourceRegime, simulate_process, simulate_sources

CONTINUOUS_FAMILIES = (
    Family.E2O,
    Family.EVENT_EXISTS,
    Family.EVENT_TYPE,
    Family.EVENT_TIME,
    Family.OBJECT_ATTR,
    Family.EVENT_ATTR,
)


def continuous_registry(n: int) -> VariableRegistry:
    reg = VariableRegistry()
    for i in range(n):
        reg.add_continuous(AssertionRef.event_time(f"e{i}"))
    return reg.freeze()


EXACT = EPConfig(max_iter=4000, tol=1e-10)
"""Run to a tolerance far below the gap being measured.

An oracle comparison must test the tilted moments, not where the default stopping rule
happened to halt; at the default tolerance a passing test would be indistinguishable from one
that stopped early in the right place.
"""


def numerical_moments(density, lo=-14.0, hi=14.0) -> tuple[float, float]:
    """First two moments of an unnormalised density, by quadrature.

    The oracle every expectation-propagation test below is measured against.
    """
    mass = integrate.quad(density, lo, hi, limit=400)[0]
    mean = integrate.quad(lambda z: z * density(z), lo, hi, limit=400)[0] / mass
    second = integrate.quad(lambda z: z * z * density(z), lo, hi, limit=400)[0] / mass
    return mean, second - mean * mean


# -- marginal transforms ----------------------------------------------------------------


def test_gaussian_marginal_round_trips():
    marginal = GaussianMarginal(5.0, 2.0)
    values = np.array([1.0, 5.0, 9.5])
    assert np.allclose(marginal.to_value(marginal.to_latent(values)), values)
    assert marginal.scale == pytest.approx(2.0)


def test_empirical_marginal_round_trips_and_extrapolates():
    """Both tails must stay invertible, or extreme values collapse onto one coordinate."""
    sample = np.random.default_rng(0).normal(3.0, 2.0, 500)
    marginal = EmpiricalMarginal.fit(sample)
    probe = np.array([-20.0, 0.0, 3.0, 25.0])
    assert np.allclose(marginal.to_value(marginal.to_latent(probe)), probe, atol=1e-6)
    assert np.all(np.diff(marginal.to_value(np.linspace(-4, 4, 40))) > 0)


def test_ordinal_marginal_censors_rather_than_pinning():
    marginal = OrdinalMarginal.fit([0, 0, 1, 1, 1, 2, 2, 3])
    lo, hi = marginal.to_latent_interval(1.0)
    assert lo < hi, "an ordinal observation constrains an interval, not a point"
    assert marginal.to_value(0.5 * (lo + hi)) == 1.0
    assert marginal.is_censored

    # The level intervals must tile the line, or some latent value maps to no level.
    edges = marginal.cutpoints[:-1]
    assert np.all(np.diff(edges) > 0)


def test_truncated_marginal_separates_the_floor_from_the_tail():
    marginal = TruncatedMarginal.fit([0.0, 0.0, 0.0, 1.5, 2.5, 3.5])
    lo, hi = marginal.to_latent_interval(0.0)
    assert lo == -np.inf and np.isfinite(hi), "the floor is censored, not observed"
    lo, hi = marginal.to_latent_interval(2.5)
    assert lo == hi, "a value above the floor pins the coordinate"
    assert marginal.to_value(hi) == pytest.approx(2.5, abs=0.3)


def test_categorical_is_refused_by_name():
    """The copula's boundary must fail loudly, not silently mis-model an unordered type."""
    spec = AttributeSpec("channel", AttributeKind.CATEGORICAL, ("web", "shop"))
    with pytest.raises(ValueError, match="discrete layer"):
        fit_marginal([0.0, 1.0], spec=spec)


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-np.inf, 0.0), (0.5, 1.5), (5.0, 5.5), (8.0, 9.0), (12.0, 12.1), (10.0, np.inf)],
)
def test_truncated_moments_match_quadrature_into_the_far_tail(lo, hi):
    """Far-tail intervals are where the naive formula silently returns near-zero variance."""
    mean, var = truncated_normal_moments(lo, hi)
    density = lambda z: np.exp(-0.5 * z * z) if lo <= z <= hi else 0.0  # noqa: E731
    want_mean, want_var = numerical_moments(density, max(lo, -20.0), min(hi, 20.0))
    assert float(mean) == pytest.approx(want_mean, rel=1e-6)
    assert float(var) == pytest.approx(want_var, rel=1e-5)


# -- bridge functions -------------------------------------------------------------------


def test_simulated_bridge_reproduces_the_closed_form():
    """The internal cross-check the simulated bridge exists to be checkable by."""
    grid, taus = bridge_curve(GaussianMarginal(), GaussianMarginal())
    closed_form = 2.0 / np.pi * np.arcsin(grid)
    assert np.abs(taus - closed_form).max() < 0.02


def test_bridge_recovers_a_planted_correlation_through_discretisation():
    """Ties shrink tau; the bridge is what puts the latent correlation back."""
    marginal = OrdinalMarginal.fit([0, 0, 1, 1, 2, 2, 3, 3])
    rng = np.random.default_rng(1)
    rho = 0.7
    z = rng.standard_normal((2, 4000))
    partner = rho * z[0] + np.sqrt(1 - rho**2) * z[1]
    tau = kendall_tau(marginal.to_value(z[0]), marginal.to_value(partner))

    assert tau < rho, "discreteness must attenuate the rank correlation"
    assert bridge_correlation(tau, marginal, marginal) == pytest.approx(rho, abs=0.06)


def test_correlations_are_clamped_below_one():
    marginal = OrdinalMarginal.fit([0, 1])
    assert abs(bridge_correlation(0.999, marginal, marginal)) < 1.0


# -- Gaussian evidence and correlation: exact banks --------------------------------------


def test_gaussian_evidence_matches_the_analytic_posterior():
    reg = continuous_registry(1)
    graph = GaussianGraph(reg, [0], [GaussianEvidenceBank.from_moments([0], [2.0], [0.5])])
    result = run_ep(graph, EXACT)

    var = 1.0 / (1.0 + 1.0 / 0.5)
    assert result.converged
    assert result.mean[0] == pytest.approx(var * 2.0 / 0.5, abs=1e-4)
    assert result.var[0] == pytest.approx(var, abs=1e-4)


def test_correlation_bank_matches_exact_bivariate_inference():
    reg = continuous_registry(2)
    rho = 0.6
    graph = GaussianGraph(
        reg,
        [0, 1],
        [
            CorrelationBank([0], [1], np.stack([coupling_precision(rho)])),
            GaussianEvidenceBank.from_moments([0], [1.5], [0.25]),
        ],
    )
    result = run_ep(graph, EPConfig(max_iter=4000, tol=1e-10, damping=0.5))

    precision = np.array([[1.0, -rho], [-rho, 1.0]]) / (1.0 - rho**2)
    precision[0, 0] += 1.0 / 0.25
    covariance = np.linalg.inv(precision)
    expected = covariance @ np.array([1.5 / 0.25, 0.0])

    assert result.converged
    assert result.mean == pytest.approx(expected, abs=1e-4)
    assert result.var == pytest.approx(np.diag(covariance), abs=1e-4)


def test_the_gaussian_block_refuses_a_discrete_variable():
    reg = VariableRegistry()
    reg.add_binary(AssertionRef.event_exists("e0"))
    reg.freeze()
    with pytest.raises(ValueError, match="continuous variables only"):
        GaussianGraph(reg, [0], [])


# -- the four expectation-propagation cases of design doc section 4.3 --------------------


def test_copula_warp_site_matches_numerical_integration():
    """Case 1: a Student-t claim seen through a monotone marginal."""
    marginal = GaussianMarginal(10.0, 3.0)
    reg = continuous_registry(1)
    value, scale, df = np.array([16.0]), np.array([2.0]), np.array([3.0])
    graph = GaussianGraph(
        reg, [0], [ObservationBank([0], student_t_loglik(value, scale, df), marginal)]
    )
    result = run_ep(graph, EXACT)

    def density(z):
        observed = marginal.mean + marginal.sd * z
        return np.exp(-0.5 * z * z) * (1.0 + ((16.0 - observed) / 2.0) ** 2 / 3.0) ** -2.0

    want_mean, want_var = numerical_moments(density)
    assert result.converged
    # Tighter than the engine's tolerance, so this measures the quadrature and not the
    # stopping rule -- which is the pairing DEFAULT_QUADRATURE is chosen to satisfy.
    assert result.mean[0] == pytest.approx(want_mean, abs=1e-5)
    assert result.var[0] == pytest.approx(want_var, abs=1e-5)


def test_a_coarser_quadrature_biases_the_tilted_moments():
    """The heavy Student-t tail is what sets the default node count."""
    marginal = GaussianMarginal(10.0, 3.0)
    reg = continuous_registry(1)
    value, scale, df = np.array([16.0]), np.array([2.0]), np.array([3.0])

    def density(z):
        observed = marginal.mean + marginal.sd * z
        return np.exp(-0.5 * z * z) * (1.0 + ((16.0 - observed) / 2.0) ** 2 / 3.0) ** -2.0

    _want_mean, want_var = numerical_moments(density)
    errors = []
    for nodes in (16, 32, 64):
        graph = GaussianGraph(
            reg,
            [0],
            [
                ObservationBank(
                    [0], student_t_loglik(value, scale, df), marginal, n_quadrature=nodes
                )
            ],
        )
        errors.append(abs(run_ep(graph, EXACT).var[0] - want_var))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < EPConfig().tol, "quadrature error must sit below the tolerance"


def test_interval_censoring_site_matches_the_closed_form():
    """Case 2: an ordinal or bracketed observation."""
    reg = continuous_registry(1)
    graph = GaussianGraph(reg, [0], [IntervalBank([0], [0.5], [1.5])])
    result = run_ep(graph, EXACT)

    want_mean, want_var = truncated_normal_moments(0.5, 1.5)
    assert result.mean[0] == pytest.approx(float(want_mean), abs=1e-4)
    assert result.var[0] == pytest.approx(float(want_var), abs=1e-4)


def test_hard_precedence_matches_the_exact_truncated_pair():
    """Case 3, in the one setting with a closed form: the exact answer is -1/sqrt(pi)."""
    reg = continuous_registry(2)
    graph = GaussianGraph(reg, [0, 1], [PrecedenceBank([0], [1], 1.0, 1.0, hard=True)])
    result = run_ep(graph, EPConfig(damping=0.7))

    assert result.mean[0] == pytest.approx(-1.0 / np.sqrt(np.pi), abs=1e-3)
    assert result.mean[1] == pytest.approx(+1.0 / np.sqrt(np.pi), abs=1e-3)


def test_soft_precedence_orders_two_timestamps():
    reg = continuous_registry(2)
    graph = GaussianGraph(reg, [0, 1], [PrecedenceBank([0], [1], 0.3, 3.0)])
    result = run_ep(graph, EPConfig(damping=0.7))

    grid = np.linspace(-6, 6, 481)
    early, late = np.meshgrid(grid, grid, indexing="ij")
    weight = np.exp(-0.5 * (early**2 + late**2)) * np.exp(
        -3.0 * np.logaddexp(0.0, -(late - early) / 0.3)
    )
    weight /= weight.sum()

    assert result.converged
    assert result.mean[0] < 0 < result.mean[1]
    assert result.mean[0] == pytest.approx(float((weight * early).sum()), abs=0.02)
    assert result.mean[1] == pytest.approx(float((weight * late).sum()), abs=0.02)


def test_cg_collapse_matches_the_exact_mixture_moments():
    """Case 4: the mixture message, against brute-force integration of the same mixture."""
    reg = continuous_registry(1)
    bank = CGMeanBank([0], np.array([[-1.0, 2.0]]), np.array([[0.5, 0.5]]), 0.25)
    result = run_ep(GaussianGraph(reg, [0], [bank]), EXACT)

    def density(z):
        return np.exp(-0.5 * z * z) * 0.5 * (
            np.exp(-0.5 * (z + 1.0) ** 2 / 0.25) + np.exp(-0.5 * (z - 2.0) ** 2 / 0.25)
        )

    want_mean, want_var = numerical_moments(density)
    assert result.mean[0] == pytest.approx(want_mean, abs=1e-4)
    assert result.var[0] == pytest.approx(want_var, abs=1e-4)


def test_the_retained_mixture_is_available_beside_its_collapse():
    """Design doc section 4.3 keeps the uncollapsed path so the collapse can be checked."""
    reg = continuous_registry(1)
    bank = CGMeanBank([0], np.array([[-1.0, 2.0]]), np.array([[0.5, 0.5]]), 0.25)
    graph = GaussianGraph(reg, [0], [bank])
    result = run_ep(graph, EXACT)

    weights, means, variances = bank.mixture(cavities(graph, result))
    collapsed_mean = float((weights[0] * means[0]).sum())
    collapsed_var = float(
        (weights[0] * (variances[0] + means[0] ** 2)).sum() - collapsed_mean**2
    )
    assert weights[0].sum() == pytest.approx(1.0)
    assert collapsed_mean == pytest.approx(result.mean[0], abs=1e-4)
    assert collapsed_var == pytest.approx(result.var[0], abs=1e-4)


def test_cg_message_to_the_discrete_side_is_the_exact_log_partition():
    """The exact direction of the coupling, checked against the convolution it claims to be."""
    bank = CGMeanBank([0], np.array([[-1.0, 2.0]]), np.array([[0.5, 0.5]]), 0.25)
    cavity = np.array([[1.0 / 0.5, 2.0 / 0.5]])  # N(2.0, 0.5)
    rows = bank.discrete_potentials(cavity)

    total = 0.25 + 0.5
    exact = np.array(
        [-0.5 * (np.log(2 * np.pi * total) + (2.0 - mu) ** 2 / total) for mu in (-1.0, 2.0)]
    )
    assert rows[0] == pytest.approx(exact - exact.max(), abs=1e-9)


def test_a_timestamp_moves_the_type_posterior_through_the_coupling():
    """The coupling is worth having only if it is bidirectional; this is the harder side."""
    bank = CGMeanBank([0], np.array([[-2.0, 2.0]]), np.array([[0.5, 0.5]]), 0.25)
    late = bank.discrete_potentials(np.array([[4.0, 8.0]]))  # cavity mean +2
    early = bank.discrete_potentials(np.array([[4.0, -8.0]]))  # cavity mean -2

    assert late[0, 1] > late[0, 0], "a late timestamp favours the late-mean type"
    assert early[0, 0] > early[0, 1]


# -- engine behaviour --------------------------------------------------------------------


def test_an_improper_cavity_is_skipped_rather_than_clipped():
    """Clipping a cavity would invent evidence; skipping only costs a sweep."""
    reg = continuous_registry(1)
    graph = GaussianGraph(
        reg, [0], [GaussianEvidenceBank([0], [0.0], [0.0]), IntervalBank([0], [-1.0], [1.0])]
    )
    result = run_ep(graph, EPConfig(max_iter=50))
    assert np.isfinite(result.mean[0]) and result.var[0] > 0


def test_non_convergence_is_reported_not_hidden():
    reg = continuous_registry(2)
    graph = GaussianGraph(reg, [0, 1], [PrecedenceBank([0], [1], 0.05, 40.0)])
    result = run_ep(graph, EPConfig(max_iter=3, damping=0.99))
    assert result.converged is False
    assert result.diagnostics()["ep_converged"] is False
    assert result.iterations == 3


def test_attribution_decomposes_the_posterior_mean_exactly():
    """The continuous analogue of the log-odds decomposition, and equally free."""
    reg = continuous_registry(1)
    graph = GaussianGraph(
        reg,
        [0],
        [
            GaussianEvidenceBank.from_moments([0], [2.0], [0.5], name="a", labels=["src_a"]),
            GaussianEvidenceBank.from_moments([0], [-1.0], [1.0], name="b", labels=["src_b"]),
        ],
    )
    result = run_ep(graph, EXACT)
    builder = BeliefStateBuilder(reg)
    fill_belief_state(builder, graph, result)
    belief = builder.build()

    ref = AssertionRef.event_time("e0")
    contributions = belief.attribution(ref)
    assert set(contributions) == {"prior", "a:src_a", "b:src_b"}
    assert sum(contributions.values()) == pytest.approx(belief.gaussian(ref)[0], abs=1e-6)


def test_gauss_hermite_weights_average_to_one():
    _nodes, weights = gauss_hermite(32)
    assert weights.sum() == pytest.approx(1.0)


def test_natural_parameters_round_trip():
    mean, var = np.array([1.5, -2.0]), np.array([0.25, 4.0])
    got_mean, got_var = to_moments(to_natural(mean, var))
    assert got_mean == pytest.approx(mean)
    assert got_var == pytest.approx(var)


# -- the belief-state contract ------------------------------------------------------------


def test_values_are_reported_in_observed_units_and_intervals_are_monotone_images():
    reg = continuous_registry(1)
    marginal = GaussianMarginal(100.0, 5.0)
    builder = BeliefStateBuilder(reg)
    builder.set_continuous(0, 0.5, 0.25, marginal=marginal)
    belief = builder.build()

    ref = AssertionRef.event_time("e0")
    assert belief.value(ref) == pytest.approx(102.5)
    lo, hi = belief.value_interval(ref, 0.9)
    assert lo < 102.5 < hi
    # A monotone transform preserves quantiles, so the observed interval is the image of the
    # latent one rather than an approximation of it.
    assert lo == pytest.approx(marginal.to_value(0.5 - 1.6448536 * 0.5), abs=1e-4)


def test_a_coordinate_without_a_marginal_refuses_to_invent_units():
    reg = continuous_registry(1)
    builder = BeliefStateBuilder(reg)
    builder.set_continuous(0, 0.5, 0.25)
    belief = builder.build()
    with pytest.raises(KeyError, match="no marginal transform"):
        belief.value(AssertionRef.event_time("e0"))


def test_crps_of_a_perfect_forecast_beats_a_vague_one():
    sharp = gaussian_crps(np.array(0.0), np.array(0.01), np.array(0.0))
    vague = gaussian_crps(np.array(0.0), np.array(4.0), np.array(0.0))
    assert sharp < vague, "a proper score must reward sharpness when the forecast is right"

    confident_and_wrong = gaussian_crps(np.array(0.0), np.array(0.01), np.array(3.0))
    assert confident_and_wrong > vague, "and punish sharpness when it is wrong"


# -- grounding --------------------------------------------------------------------------


def test_a_universe_without_attributes_grounds_no_attribute_variables():
    """The layer must be additive: declaring nothing continuous must cost nothing."""
    truth = simulate_process(ProcessConfig(n_orders=3, seed=0))
    universe = truth.build_universe()
    refs = continuous_refs(universe)
    assert refs, "timestamps are always registered"
    assert all(r.family is Family.EVENT_TIME for r in refs)


def test_event_attributes_need_every_type_in_the_support_to_declare_them():
    """OCEL 2.0 makes attribute sets disjoint per type, so a wide support is always ambiguous."""
    truth = simulate_process(
        ProcessConfig(n_orders=3, attributes=True, p_type_known=0.5, seed=0)
    )
    universe = truth.build_universe()
    report = universe.prune_report

    assert report.attr_registered > 0
    assert report.attr_ambiguous > 0
    for ref in continuous_refs(universe):
        if ref.family is Family.EVENT_ATTR:
            support = universe.events[ref.subject].type_support
            declaring = universe.schema.event_types_declaring(ref.attribute)
            assert support <= declaring


def test_the_copula_recovers_the_planted_object_attribute_correlation():
    """The generator plants a known latent correlation; the fit must find it."""
    config = ProcessConfig(
        n_orders=60, attributes=True, attribute_correlation=0.6, seed=11
    )
    truth = simulate_process(config)
    universe = truth.build_universe()
    sources = simulate_sources(
        universe,
        truth,
        SourceRegime(n_sources=400, n_hotspots=40, deg_s_min=30, seed=11,
                     families=(Family.OBJECT_ATTR,)),
    )
    claims = ClaimSet.from_sources(sources.sources)
    copula = fit_copula_from_claims(universe, claims, seed=11)

    recovered = copula.correlation("order_value", "order_units")
    assert recovered > 0.2, "the planted correlation must survive noise and discretisation"
    assert recovered < 1.0


# -- end to end ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def continuous_run():
    """A world where the link layer is informed, so every continuous factor can ground."""
    truth = simulate_process(
        ProcessConfig(n_orders=8, attributes=True, p_type_known=0.5, seed=5)
    )
    universe = truth.build_universe()
    sources = simulate_sources(
        universe,
        truth,
        SourceRegime(
            n_sources=600, n_hotspots=60, deg_s_min=40, seed=5, families=CONTINUOUS_FAMILIES
        ),
    )
    claims = ClaimSet.from_sources(sources.sources)
    result = fuse(
        universe,
        sources.sources,
        FusionConfig(outer_iterations=2, bp=BPConfig(max_iter=100), seed=5),
    )
    return truth, universe, claims, result


def test_every_copula_kind_is_exercised(continuous_run):
    _truth, _universe, _claims, result = continuous_run
    kinds = set(result.continuous.summary()["kinds"])
    assert kinds == {"continuous", "count", "ordinal", "truncated"}


def test_all_six_continuous_templates_ground(continuous_run):
    _truth, _universe, _claims, result = continuous_run
    banks = result.continuous.summary()["banks"]
    assert set(banks) >= {
        "time_bracket",
        "continuous_channel",
        "copula_correlation",
        "cg_time",
        "precedence",
    }
    assert all(count > 0 for count in banks.values())


def test_the_continuous_layer_is_better_calibrated_than_its_baselines(continuous_run):
    """Design doc section 8.3's bar, applied to the continuous half.

    Calibration is the criterion rather than point error, per Stage 1 section 3.2: an
    aggregator that reports a median with a standard error is often close on the point and
    badly overconfident about it, which is exactly the failure fusion is supposed to fix.
    """
    truth, universe, claims, result = continuous_run
    block = frozenset(continuous_refs(universe))
    scored = [r for r in claims.refs if r in block]
    channels = pairwise_channels(
        claims, lambda ref: marginal_key(ref) if ref in block else None
    )
    copula = result.continuous.copula

    table = compare_continuous(
        {
            "claim_median": claim_median(universe.registry, claims, copula, channels=channels),
            "weighted_mean": weighted_mean(universe.registry, claims, copula, channels=channels),
            "ocbf": result.belief,
        },
        truth.truth_map(scored),
        scored,
    )
    assert table["ocbf"]["crps"] < table["claim_median"]["crps"]
    assert table["ocbf"]["crps"] < table["weighted_mean"]["crps"]
    assert abs(table["ocbf"]["coverage_error"]) < abs(table["claim_median"]["coverage_error"])
    assert abs(table["ocbf"]["coverage_error"]) < abs(table["weighted_mean"]["coverage_error"])


@pytest.fixture(scope="module")
def sparse_continuous_run():
    """A pool thin enough to leave real timestamps unspoken for.

    A separate fixture rather than a knob on the dense one, because the two regimes exercise
    opposite halves of the layer: dense coverage is what lets precedence and the coupling
    ground at all, and thin coverage is what leaves assertions for the structural prior to
    carry. One configuration cannot show both.
    """
    truth = simulate_process(
        ProcessConfig(n_orders=8, attributes=True, p_type_known=0.5, seed=5)
    )
    universe = truth.build_universe()
    sources = simulate_sources(
        universe,
        truth,
        SourceRegime(n_sources=120, n_hotspots=6, seed=5, families=CONTINUOUS_FAMILIES),
    )
    claims = ClaimSet.from_sources(sources.sources)
    result = fuse(
        universe,
        sources.sources,
        FusionConfig(outer_iterations=1, bp=BPConfig(max_iter=100), seed=5),
    )
    return truth, universe, claims, result


def test_timestamps_no_source_mentioned_still_get_a_calibrated_posterior(
    sparse_continuous_run,
):
    """The structural prior on the continuous side, which is the point of the whole layer.

    No claim, so no aggregation rule has anything to say about these at all. The bracket, the
    type-conditioned mean and the precedence orderings do, and the resulting intervals still
    have to cover.
    """
    truth, universe, claims, result = sparse_continuous_run
    silent = [
        ref
        for ref in continuous_refs(universe)
        if ref.family is Family.EVENT_TIME
        and not claims.for_ref(ref)
        and truth.truth(ref) is not None
    ]
    assert len(silent) > 10, "the thin pool must leave real timestamps unclaimed"

    report = evaluate_continuous(result.belief, truth.truth_map(silent), silent)
    assert report.n == len(silent)
    assert report.coverage > 0.6


def test_the_posterior_is_probability_calibrated(continuous_run):
    """Probability-integral transforms are uniform exactly when the posterior is honest."""
    truth, universe, claims, result = continuous_run
    block = frozenset(continuous_refs(universe))
    scored = [r for r in claims.refs if r in block]
    values = pit_values(result.belief, truth.truth_map(scored), scored)

    assert values.size > 30
    assert 0.35 < float(values.mean()) < 0.65
    # Mass piled at both ends is the signature of intervals that are too narrow.
    extreme = float(np.mean((values < 0.05) | (values > 0.95)))
    assert extreme < 0.30


def test_the_continuous_layer_leaves_a_discrete_only_world_untouched():
    """The layer is additive, and this is the guarantee the default depends on."""
    truth = simulate_process(ProcessConfig(n_orders=5, seed=2))
    universe = truth.build_universe()
    sources = simulate_sources(
        universe, truth, SourceRegime(n_sources=120, n_hotspots=10, seed=2)
    ).sources

    config = FusionConfig(outer_iterations=1, bp=BPConfig(max_iter=60), fit_reliability=False)
    with_layer = fuse(universe, sources, config)
    config.continuous = None
    without_layer = fuse(universe, sources, config)

    refs = [r for r in with_layer.claim_set.refs if r.family is Family.E2O]
    assert np.allclose(
        with_layer.belief.binary_scores(refs), without_layer.belief.binary_scores(refs)
    )


def test_the_hybrid_loop_feeds_the_continuous_message_back_to_the_discrete_graph(
    continuous_run,
):
    """The exact half of the coupling has to actually reach the discrete backbone."""
    _truth, universe, _claims, result = continuous_run
    grounding = result.continuous
    assert grounding.cg_targets, "some event must have both a latent type and a timestamp"

    for position, target in grounding.cg_targets.items():
        assert universe.registry.ref(target).family is Family.EVENT_TYPE
        assert grounding.graph.registry.ref(
            int(grounding.graph.var_ids[position])
        ).family is Family.EVENT_TIME


def test_precedence_can_be_promoted_to_a_hard_constraint(continuous_run):
    """The constraint register governs this factor exactly as it governs the discrete ones."""
    _truth, universe, claims, result = continuous_run
    from ocbf.schema import ConstraintRegister

    register = ConstraintRegister().override(
        ConstraintClass.LIFECYCLE_PRECEDENCE, strength=Strength.HARD
    )
    grounding = build_continuous_graph(
        universe,
        claims,
        ReliabilityTable(),
        copula=result.continuous.copula,
        discrete=result.belief,
        register=register,
        seed=5,
    )
    assert grounding.graph.summary()["banks"].get("precedence", 0) > 0


def test_disabling_precedence_is_reported_in_the_grounding(continuous_run):
    _truth, universe, claims, result = continuous_run
    grounding = build_continuous_graph(
        universe,
        claims,
        ReliabilityTable(),
        copula=result.continuous.copula,
        discrete=result.belief,
        spec=ContinuousSpec(precedence=False),
        seed=5,
    )
    assert "precedence" not in grounding.graph.summary()["banks"]


# -- behaviour: what the layer does, rather than how it computes it -----------------------


def test_evidence_moves_a_timestamp_toward_what_the_sources_said():
    marginal = GaussianMarginal(100.0, 10.0)
    reg = continuous_registry(1)
    claims = np.array([118.0, 121.0, 119.0])
    graph = GaussianGraph(
        reg,
        [0],
        [
            ObservationBank(
                np.zeros(3, dtype=np.int64),
                student_t_loglik(claims, np.full(3, 4.0), np.full(3, 3.0)),
                marginal,
            )
        ],
    )
    builder = BeliefStateBuilder(reg)
    fill_belief_state(builder, graph, run_ep(graph), CopulaSpec({"event_time": marginal}))
    belief = builder.build()

    assert belief.value(AssertionRef.event_time("e0")) == pytest.approx(119.0, abs=2.0)


def test_agreeing_sources_sharpen_the_posterior():
    """Sharpening with evidence is the property that makes a stated interval worth reading."""
    marginal = GaussianMarginal(0.0, 1.0)
    reg = continuous_registry(1)
    widths = []
    for count in (1, 3, 9):
        claims = np.full(count, 1.0)
        graph = GaussianGraph(
            reg,
            [0],
            [
                ObservationBank(
                    np.zeros(count, dtype=np.int64),
                    student_t_loglik(claims, np.full(count, 0.5), np.full(count, 3.0)),
                    marginal,
                )
            ],
        )
        widths.append(run_ep(graph, EXACT).var[0])

    assert widths[0] > widths[1] > widths[2]


def test_the_student_t_channel_resists_an_outlier_where_a_gaussian_one_does_not():
    """The reason design doc section 5.1 chooses Student-t, stated as a measurable difference."""
    marginal = GaussianMarginal(0.0, 1.0)
    reg = continuous_registry(1)
    honest = np.array([0.2, 0.3, 0.25])
    with_outlier = np.array([0.2, 0.3, 0.25, 8.0])

    def posterior(values, channel):
        scale = np.full(len(values), 0.4)
        loglik = (
            channel(values, scale, np.full(len(values), 3.0))
            if channel is student_t_loglik
            else channel(values, scale)
        )
        graph = GaussianGraph(
            reg,
            [0],
            [ObservationBank(np.zeros(len(values), dtype=np.int64), loglik, marginal)],
        )
        return run_ep(graph, EXACT).mean[0]

    robust = abs(
        posterior(with_outlier, student_t_loglik) - posterior(honest, student_t_loglik)
    )
    naive = abs(
        posterior(with_outlier, gaussian_loglik) - posterior(honest, gaussian_loglik)
    )
    assert robust < 0.5 * naive


def test_a_correlated_attribute_borrows_strength_from_its_partner():
    """What the copula's precision structure buys, and the only reason to estimate it."""
    reg = VariableRegistry()
    for name in ("order_value", "order_units"):
        reg.add_continuous(AssertionRef.object_attr("order_0", name))
    reg.freeze()
    observed, unobserved = 0, 1
    evidence = GaussianEvidenceBank.from_moments([observed], [2.0], [0.1])

    without = run_ep(GaussianGraph(reg, [0, 1], [evidence]), EXACT)
    with_link = run_ep(
        GaussianGraph(
            reg,
            [0, 1],
            [evidence, CorrelationBank([0], [1], np.stack([coupling_precision(0.7)]))],
        ),
        EXACT,
    )

    assert without.mean[unobserved] == pytest.approx(0.0, abs=1e-6)
    assert with_link.mean[unobserved] > 0.5, "evidence must travel along a fitted correlation"
    assert with_link.var[unobserved] < without.var[unobserved]


def test_an_uncorrelated_attribute_borrows_nothing():
    """The other half of the same contract: a zero in the precision is a claim of independence."""
    reg = VariableRegistry()
    for name in ("order_value", "item_weight"):
        reg.add_continuous(AssertionRef.object_attr("order_0", name))
    reg.freeze()
    graph = GaussianGraph(
        reg,
        [0, 1],
        [
            GaussianEvidenceBank.from_moments([0], [2.0], [0.1]),
            CorrelationBank([0], [1], np.stack([coupling_precision(0.0)])),
        ],
    )
    result = run_ep(graph, EXACT)
    assert result.mean[1] == pytest.approx(0.0, abs=1e-6)
    assert result.var[1] == pytest.approx(1.0, abs=1e-6)


def test_precedence_orders_the_events_it_is_grounded_on(continuous_run):
    """The factor has to change the answer, not merely be present in the bank list."""
    _truth, universe, claims, result = continuous_run
    grounding = build_continuous_graph(
        universe,
        claims,
        ReliabilityTable(),
        copula=result.continuous.copula,
        discrete=result.belief,
        spec=ContinuousSpec(precedence_slack=0.5),
        seed=5,
    )
    bank_index = next(
        i for i, b in enumerate(grounding.graph.banks) if b.name == "precedence"
    )
    bank = grounding.graph.banks[bank_index]
    edges = bank.edge_vars()
    half = len(edges) // 2

    with_factor = run_ep(grounding.graph, EPConfig(max_iter=400))
    without = run_ep(
        GaussianGraph(
            grounding.graph.registry,
            grounding.graph.var_ids,
            [b for i, b in enumerate(grounding.graph.banks) if i != bank_index],
        ),
        EPConfig(max_iter=400),
    )

    def ordered(means):
        return float(np.mean(means[edges[:half]] < means[edges[half:]]))

    assert ordered(with_factor.mean) >= ordered(without.mean)


def test_a_misstated_lifecycle_is_overruled_when_soft_and_obeyed_when_hard():
    """Design doc section 8.3's sensitivity experiment, on the continuous side.

    The whole point of registering precedence SOFT is that evidence can win. Asserting the
    order backwards against two confident, clearly-ordered claims must therefore leave the
    claims in charge -- while the same constraint made HARD overturns them, which is the
    damage a wrong definitional rule does.
    """
    marginal = GaussianMarginal(0.0, 1.0)
    reg = continuous_registry(2)
    claims = ObservationBank(
        np.array([0, 1]),
        student_t_loglik(np.array([1.5, -1.5]), np.array([0.3, 0.3]), np.array([3.0, 3.0])),
        marginal,
    )
    # The claims say variable 0 is late; the constraint insists it comes first.
    soft = run_ep(
        GaussianGraph(reg, [0, 1], [claims, PrecedenceBank([0], [1], 0.5, 1.0)]),
        EPConfig(max_iter=400, damping=0.9),
    )
    hard = run_ep(
        GaussianGraph(reg, [0, 1], [claims, PrecedenceBank([0], [1], 0.5, 1.0, hard=True)]),
        EPConfig(max_iter=400, damping=0.9),
    )

    assert soft.mean[0] > soft.mean[1], "a soft constraint must yield to good evidence"
    assert hard.mean[0] < hard.mean[1], "a hard one overrides it, which is the risk it carries"


def test_a_decoy_event_keeps_its_bracket_prior(continuous_run):
    """An assertion nobody can speak about must fall back, not invent a confident answer."""
    truth, universe, claims, result = continuous_run
    decoys = [
        ref
        for ref in continuous_refs(universe)
        if ref.family is Family.EVENT_TIME and truth.truth(ref) is None
    ]
    assert decoys

    for ref in decoys[:5]:
        mean, var = result.belief.gaussian(ref)
        assert np.isfinite(mean) and var > 0
        lo, hi = result.belief.value_interval(ref, 0.9)
        event = universe.events[ref.subject]
        assert lo >= event.time_lo - 3.0 and hi <= event.time_hi + 3.0
