"""Explicit public API groups shared by generation and documentation validation."""

import ast
from pathlib import Path

GROUPS = {
    "Workflow": ("ocbf.api",),
    "Semantic context": (
        "ocbf",
        "ocbf.schema",
        "ocbf.schema.core",
        "ocbf.schema.constraints",
        "ocbf.assertions",
        "ocbf.assertions.refs",
        "ocbf.assertions.registry",
        "ocbf.universe",
        "ocbf.universe.core",
        "ocbf.universe.context",
    ),
    "Evidence and parameters": (
        "ocbf.evidence",
        "ocbf.evidence.records",
        "ocbf.evidence.snapshots",
        "ocbf.evidence.observations",
        "ocbf.sources.contracts",
        "ocbf.sources.interpretation",
        "ocbf.reliability.config",
        "ocbf.reliability.channels",
    ),
    "Canonical model": (
        "ocbf.model.spec",
        "ocbf.model.compile",
        "ocbf.model.contracts",
        "ocbf.model.channels",
        "ocbf.model.kernels",
        "ocbf.model.hybrid_kernels",
        "ocbf.model.reductions",
        "ocbf.model.decoding",
    ),
    "Inference and belief": (
        "ocbf.inference.contracts",
        "ocbf.inference.router",
        "ocbf.inference.registry",
        "ocbf.inference.adapters.exact",
        "ocbf.inference.adapters.hybrid",
        "ocbf.inference.adapters.approximate",
        "ocbf.inference.elimination",
        "ocbf.inference.sampling",
        "ocbf.inference.warm_start",
        "ocbf.belief.posterior",
        "ocbf.belief.estimates",
        "ocbf.belief.samples",
        "ocbf.diagnostics.monte_carlo",
    ),
    "Queries": (
        "ocbf.queries",
        "ocbf.queries.expressions",
        "ocbf.queries.contracts",
        "ocbf.queries.evaluate",
    ),
    "Execution and interchange": (
        "ocbf.runtime.contracts",
        "ocbf.runtime.control",
        "ocbf.runtime.session",
        "ocbf.runtime.cache",
        "ocbf.model.dependencies",
        "ocbf.io",
        "ocbf.io.bundle",
        "ocbf.io.arrays",
        "ocbf.errors",
        "ocbf.backends",
    ),
    "Numerical utilities": (
        "ocbf.inference",
        "ocbf.inference.loopy_bp",
        "ocbf.inference.gabp_ep",
        "ocbf.inference.gtsam_exact",
        "ocbf.model",
        "ocbf.model.graph",
        "ocbf.model.banks",
        "ocbf.model.gaussian",
        "ocbf.model.gaussian_banks",
        "ocbf.model.copula",
        "ocbf.model.copula.marginals",
        "ocbf.model.copula.bridge",
        "ocbf.model.copula.structure",
        "ocbf.belief",
        "ocbf.belief.state",
        "ocbf.mathx",
    ),
    "Source diagnostics and evaluation": (
        "ocbf.sources",
        "ocbf.sources.base",
        "ocbf.sources.claims",
        "ocbf.reliability",
        "ocbf.reliability.params",
        "ocbf.reliability.moments",
        "ocbf.diagnostics",
        "ocbf.diagnostics.decidability",
        "ocbf.diagnostics.ess",
        "ocbf.diagnostics.overlap",
        "ocbf.baselines",
        "ocbf.baselines.vote",
        "ocbf.baselines.dawid_skene",
        "ocbf.baselines.aggregate",
        "ocbf.eval",
        "ocbf.eval.metrics",
        "ocbf.eval.continuous",
        "ocbf.synth",
        "ocbf.synth.process",
        "ocbf.synth.corrupt",
    ),
}


# Navigation tiers: the facade and scientific contracts first, narrower utilities after.
TIERS = {
    "Workflow and contracts": (
        "Workflow",
        "Semantic context",
        "Evidence and parameters",
        "Canonical model",
        "Inference and belief",
        "Queries",
        "Execution and interchange",
    ),
    "Numerical and source utilities": (
        "Numerical utilities",
        "Source diagnostics and evaluation",
    ),
}


def tier_of(group):
    """Return the navigation tier that contains a reference group."""
    return next(tier for tier, groups in TIERS.items() if group in groups)


def module_path(root, module):
    """Resolve only an explicit public package/module under the repository root."""
    if any(part.startswith("_") for part in module.split(".")):
        raise ValueError(f"private module in public reference: {module}")
    path = Path(root).joinpath(*module.split("."))
    source = path.with_suffix(".py")
    if not source.is_file():
        source = path / "__init__.py"
    if not source.is_file():
        raise ValueError(f"reference module does not exist: {module}")
    return source


def validate_inventory(root):
    """Validate module identities and explicit exports without importing implementations."""
    seen = set()
    for group, modules in GROUPS.items():
        for module in modules:
            if module in seen:
                raise ValueError(f"duplicate reference module: {module}")
            seen.add(module)
            source = module_path(root, module)
            tree = ast.parse(source.read_text(encoding="utf-8"))
            bound = set()
            exports = None
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    bound.add(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    bound.update(a.asname or a.name.split(".")[0] for a in node.names)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                    names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                    bound.update(names)
                    if "__all__" in names:
                        exports = ast.literal_eval(node.value)
            if exports is not None and (missing := set(exports) - bound):
                raise ValueError(f"unbound public exports in {module}: {sorted(missing)}")
            yield group, module, source
