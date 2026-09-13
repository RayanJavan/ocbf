"""Explicit built-in composition; implementations load during planning/inference only."""


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
