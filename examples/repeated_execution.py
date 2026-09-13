"""Reproducible synthetic workloads for repeated execution and cache measurements.

Use ``--baseline`` to record fresh-only measurements, or run without that flag for paired
fresh/cold/warm process measurements and same-target checks. Inputs are synthetic;
the benchmark does not establish factory calibration or universal speedups.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import scipy

from examples.joint_inference import inputs
from ocbf._values import portable
from ocbf.api import compile_model, evaluate, infer, requirements_for
from ocbf.inference.adapters.exact import ExactEngine
from ocbf.inference.contracts import InferencePolicy
from ocbf.io import dumps
from ocbf.queries import QueryBundle

PROTOCOL = {
    "version": "reuse-workloads-v1",
    "source_status": "synthetic numerical fixtures; no factory accuracy claim",
    "seed": 20260912,
    "repetitions": 5,
    "iterations": 12,
    "cache_bytes": 64 * 1024 * 1024,
    "exact_atol": 1e-10,
    "exact_rtol": 1e-9,
    "performance_gate": "warm median + MAD < fresh median - MAD for both workloads",
    "workloads": ["trust", "queries"],
    "engine": "reference_elimination",
}


def array_bytes(value):
    """Logical retained array payload bytes, distinct from cache charge and peak RSS."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(array_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(v) for v in value)
    if is_dataclass(value):
        return sum(array_bytes(getattr(value, f.name)) for f in fields(value) if f.init)
    return 0


def peak_rss():
    """OS process high-water mark; includes native arrays, not just Python allocations."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in ("peak", "working", "pp", "p", "np", "n", "page", "peakpage")
            ]

        values = Counters()
        values.cb = ctypes.sizeof(values)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(values), values.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(values.peak)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def worker(workload, mode):
    # All modes warm imports and use exactly the same scientific input sequence.
    started = time.perf_counter()
    spec, queries, settings, _ = inputs()
    prepare_seconds = time.perf_counter() - started
    requirements = requirements_for(queries)
    policy = InferencePolicy(engine=PROTOCOL["engine"])
    session = None
    if mode != "fresh":
        from ocbf.api import ExecutionSession

        session = ExecutionSession(max_cache_bytes=PROTOCOL["cache_bytes"])
    kwargs = {} if session is None else {"session": session}
    phases = {name: 0.0 for name in ("compile", "plan", "infer", "query", "export")}
    answers = []
    retained, scientific = {}, []

    class MeasuredEngine:
        # Instrument the existing assessment; never plan twice to obtain phase timings.
        name, version = PROTOCOL["engine"], "1"
        delegate = ExactEngine(name)

        def assess_with_context(self, *args, **kw):
            return timed("plan", lambda: self.delegate.assess_with_context(*args, **kw))

        def solve_with_context(self, *args, **kw):
            return self.delegate.solve_with_context(*args, **kw)

    engines = {PROTOCOL["engine"]: MeasuredEngine()}

    def timed(name, fn):
        before = time.perf_counter()
        value = fn()
        phases[name] += time.perf_counter() - before
        return value

    base = compile_model(spec)
    posterior = infer(base, requirements=requirements, policy=policy)

    def iteration(index):
        if workload == "trust":
            for name, parameters in sorted(settings.items()):
                model = timed(
                    "compile",
                    lambda parameters=parameters: compile_model(
                        replace(spec, parameters=parameters), **kwargs
                    ),
                )
                result = timed(
                    "infer",
                    lambda model=model: infer(
                        model, requirements=requirements, policy=policy, engines=engines, **kwargs
                    ),
                )
                answer = timed("query", lambda result=result: evaluate(result, queries, **kwargs))
                answers.append(
                    (
                        name,
                        tuple((e.value, e.denominator, dict(e.outcomes)) for e in answer.estimates),
                    )
                )
                scientific.append(answer.estimates)
                retained[name] = result
        else:
            # Threshold changes must recompute outcomes, including pending states.
            changed = QueryBundle(
                tuple(
                    replace(
                        q,
                        reference={
                            "reference_id": "reuse-threshold",
                            "threshold_minutes": {"a": 20 + index % 3, "b": 30},
                        },
                    )
                    for q in queries.queries
                )
            )
            answer = timed("query", lambda: evaluate(posterior, changed, **kwargs))
            answers.append(
                tuple((e.value, e.denominator, dict(e.outcomes)) for e in answer.estimates)
            )
            scientific.append(answer.estimates)
        timed("export", lambda: dumps(answer))

    if mode == "warm":
        for i in range(3):
            iteration(i)
        phases = dict.fromkeys(phases, 0.0)
        answers.clear()
        scientific.clear()
    started = time.perf_counter()
    for i in range(PROTOCOL["iterations"]):
        iteration(i)
    elapsed = time.perf_counter() - started
    phases["infer"] -= phases["plan"]
    output = {
        "workload": workload,
        "mode": mode,
        "seconds": elapsed,
        "phases_seconds": {"prepare_inputs": prepare_seconds, **phases},
        "peak_rss_bytes": peak_rss(),
        "pid": os.getpid(),
        "answers": answers,
        "variables": len(base.variables),
        "factors": len(base.factors),
        "cache": dict(session.stats) if session else None,
        "retained_array_bytes": array_bytes(retained or posterior),
        "scientific_estimates": portable(scientific),
        "policy": portable(policy),
        "engine_version": "1",
        "setup": {"model_id": base.model_id, "query_bundle_id": queries.bundle_id},
    }
    if session:
        session.close()
    return output


def run(output, baseline=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    modes = ("fresh",) if baseline else ("fresh", "cold", "warm")
    for repeat in range(PROTOCOL["repetitions"]):
        for workload in PROTOCOL["workloads"]:
            for mode in modes if repeat % 2 == 0 else tuple(reversed(modes)):
                process_started = time.perf_counter()
                completed = subprocess.run(
                    [sys.executable, "-m", "examples.repeated_execution", "--worker", workload, "--mode", mode],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                records.append(
                    {
                        "repeat": repeat,
                        "process_seconds": time.perf_counter() - process_started,
                        **json.loads(completed.stdout),
                    }
                )
    summary = {}
    for workload in PROTOCOL["workloads"]:
        summary[workload] = {}
        reference = next(r["answers"] for r in records if r["workload"] == workload)
        scientific_reference = next(
            r["scientific_estimates"] for r in records if r["workload"] == workload
        )
        for mode in modes:
            rows = [r for r in records if r["workload"] == workload and r["mode"] == mode]
            values = [r["seconds"] for r in rows]
            median = statistics.median(values)
            summary[workload][mode] = {
                "median_seconds": median,
                "mad_seconds": statistics.median(abs(v - median) for v in values),
                "peak_rss_bytes": max(r["peak_rss_bytes"] for r in rows),
                "same_answers": all(r["answers"] == reference for r in rows),
                "same_full_estimates": all(
                    r["scientific_estimates"] == scientific_reference for r in rows
                ),
            }
        if not baseline:
            fresh, warm = summary[workload]["fresh"], summary[workload]["warm"]
            summary[workload]["benefit"] = (
                warm["median_seconds"] + warm["mad_seconds"]
                < fresh["median_seconds"] - fresh["mad_seconds"]
            )
    report = {
        "protocol": PROTOCOL,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "records": records,
        "summary": summary,
        "measurement_revision": "2: separated existing planning time; full distributions and qualifications; retained arrays",
    }
    destination = output / ("baseline.json" if baseline else "comparison.json")
    if baseline and destination.exists():
        raise ValueError("baseline already exists; select a new output directory")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not baseline and any(
        not row["benefit"] or any(not row[mode]["same_full_estimates"] for mode in modes)
        for row in summary.values()
    ):
        raise RuntimeError("priority performance or same-target acceptance gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", default="artifacts/repeated-execution")
    parser.add_argument("--worker", choices=PROTOCOL["workloads"])
    parser.add_argument("--mode", choices=("fresh", "cold", "warm"), default="fresh")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.mode)))
    else:
        run(args.output, args.baseline)
