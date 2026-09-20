"""Auto route and built-in engine set; implementations load lazily inside builtin_engines."""

# Route tried in order when ``policy.engine == "auto"``: the first engine compatible within
# the declared budgets wins. ``gtsam_exact`` (optional native backend) and ``bp`` (approximate
# single-variable marginals) stay out of the route and are reachable only by explicit name.
AUTO_ROUTE = ("reference_elimination", "reference_hybrid", "blocked")


def builtin_engines():
    from .adapters.approximate import BPEngine
    from .adapters.exact import ExactEngine
    from .adapters.hybrid import HybridEngine
    from .sampling import BlockedEngine

    return {
        e.name: e
        for e in (
            ExactEngine("gtsam_exact"),
            ExactEngine("reference_elimination"),
            HybridEngine(),
            BlockedEngine(),
            BPEngine(),
        )
    }
