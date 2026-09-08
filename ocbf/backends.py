"""Optional native backends, and the compute they can actually reach.

Everything here exists because of one Windows fact: a CUDA-enabled extension does not carry
the CUDA runtime with it. GTSAM's ``gtsam.dll`` imports ``cusolver64_12`` and
``cusparse64_12`` by name, the loader looks for them on the *DLL search path* rather than on
``PATH``, and the CUDA toolkit installs them under ``bin\\x64``. A plain ``import gtsam``
therefore fails with a bare "DLL load failed" that names nothing and suggests nothing.

So the import is wrapped exactly once, here, and every consumer goes through
[`import_gtsam`][ocbf.backends.import_gtsam]. Three properties follow, and they are why
this is a module rather than two lines in a notebook:

* **Predictable** -- one search order, stated in [`cuda_library_dirs`][ocbf.backends.cuda_library_dirs], used everywhere.
* **Diagnosable** -- a failure reports the directories it searched, so the answer is a path
  to fix rather than a hex address.
* **Honest** -- [`gtsam_backend`][ocbf.backends.gtsam_backend] reports what the build can do and what the machine has,
  separately. GTSAM's CUDA acceleration lives in its *nonlinear least-squares* solvers; the
  discrete elimination [`ocbf.inference.gtsam_exact`][ocbf.inference.gtsam_exact] uses as an oracle runs on the CPU.
  Reporting a CUDA device is not a claim that the oracle uses one.

Nothing here is on any critical path, and importing this module loads no native library,
queries no driver and touches no filesystem.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import ModuleType

__all__ = [
    "BackendUnavailable",
    "CUDA_BIN_ENV",
    "CudaDevice",
    "GtsamBackend",
    "cuda_devices",
    "cuda_library_dirs",
    "enable_cuda_library_search",
    "gtsam_backend",
    "import_gtsam",
    "report",
]

CUDA_BIN_ENV = "OCBF_CUDA_BIN"
"""Environment variable that overrides CUDA library discovery entirely.

Set it to one or more directories (``os.pathsep``-separated) holding the CUDA runtime
libraries. An override is used verbatim and *stops* the search: a machine with an unusual
toolkit layout should be able to say so once, without this module second-guessing it.
"""

_CUDA_ROOT_ENV = ("CUDA_PATH", "CUDA_HOME")
_CUDA_BIN_SUBDIRS = ("bin/x64", "bin", "lib/x64")
_WINDOWS_TOOLKIT_GLOB = "NVIDIA GPU Computing Toolkit/CUDA/v*"

# ``os.add_dll_directory`` returns a handle whose ``close()`` undoes the addition. Holding
# the handles here is what keeps the search path alive for the process, and what makes a
# second call a no-op instead of a duplicate.
_added_dirs: dict[Path, object] = {}


class BackendUnavailable(ImportError):
    """An optional backend could not be loaded, with the reason spelled out.

    Subclasses `ImportError`, so ``except ImportError`` -- the idiom every optional
    dependency is already guarded by -- keeps working.
    """


@dataclass(frozen=True, slots=True)
class CudaDevice:
    """One visible GPU, as the CUDA driver describes it."""

    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int

    @property
    def architecture(self) -> str:
        """The two-digit form that ``nvcc -arch`` and ``sm_XX`` use."""
        major, minor = self.compute_capability
        return f"{major}{minor}"

    def summary(self) -> str:
        gib = self.total_memory_bytes / (1 << 30)
        return f"{self.name} (sm_{self.architecture}, {gib:.1f} GiB)"


@dataclass(frozen=True, slots=True)
class GtsamBackend:
    """What the installed GTSAM is, and what it has to run on.

    `cuda_built` and `devices` are kept apart deliberately: a GTSAM without CUDA support on
    a machine with a GPU, and a CUDA-enabled GTSAM on a machine without one, are different
    problems, and one boolean would hide which of them you have.
    """

    available: bool
    version: str | None = None
    cuda_built: bool = False
    devices: tuple[CudaDevice, ...] = ()
    searched: tuple[Path, ...] = ()
    reason: str | None = None
    module: ModuleType | None = field(default=None, repr=False, compare=False)

    @property
    def cuda_usable(self) -> bool:
        """A CUDA-enabled build *and* a device to run it on."""
        return self.cuda_built and bool(self.devices)

    def summary(self) -> str:
        if not self.available:
            return f"unavailable -- {self.reason}"
        cuda = "CUDA build" if self.cuda_built else "no CUDA build"
        device = self.devices[0].summary() if self.devices else "no CUDA device"
        return f"{self.version} ({cuda}), {device}"


def cuda_library_dirs() -> tuple[Path, ...]:
    """Directories that may hold the CUDA runtime libraries, in search order.

    The order is: the [`CUDA_BIN_ENV`][ocbf.backends.CUDA_BIN_ENV] override; then the toolkit named by ``CUDA_PATH`` or
    ``CUDA_HOME``; then any toolkit under the standard Windows install root, newest first.
    Within a toolkit, ``bin/x64`` precedes ``bin`` because CUDA 13 moved the redistributable
    DLLs there and left ``bin`` holding the tools.

    Only directories that exist are returned, so an empty result means "found nothing"
    rather than "did not look".
    """
    override = os.environ.get(CUDA_BIN_ENV, "").strip()
    if override:
        return tuple(
            p for p in (Path(part) for part in override.split(os.pathsep) if part) if p.is_dir()
        )

    roots: list[Path] = []
    for var in _CUDA_ROOT_ENV:
        value = os.environ.get(var, "").strip()
        if value:
            roots.append(Path(value))
    if sys.platform == "win32":
        for base in ("ProgramFiles", "ProgramW6432"):
            program_files = os.environ.get(base, "").strip()
            if program_files:
                roots.extend(sorted(Path(program_files).glob(_WINDOWS_TOOLKIT_GLOB), reverse=True))

    found: list[Path] = []
    for root in roots:
        for sub in _CUDA_BIN_SUBDIRS:
            candidate = root / sub
            if candidate.is_dir() and candidate not in found:
                found.append(candidate)
    return tuple(found)


def enable_cuda_library_search() -> tuple[Path, ...]:
    """Make the CUDA runtime resolvable for the rest of this process, and say from where.

    Idempotent, and a no-op anywhere but Windows: ELF and Mach-O extensions record their
    dependency search paths at link time, so there is nothing for this to fix there.
    """
    if sys.platform != "win32":
        return ()
    for directory in cuda_library_dirs():
        if directory not in _added_dirs:
            try:
                _added_dirs[directory] = os.add_dll_directory(str(directory))
            except OSError:  # pragma: no cover -- a directory that vanished mid-search
                continue
    return tuple(_added_dirs)


@cache
def cuda_devices() -> tuple[CudaDevice, ...]:
    """Every CUDA device the driver can see, or ``()`` if there is no usable driver.

    Asks the driver library directly rather than shelling out to ``nvidia-smi`` or importing
    a GPU framework, because this has to be answerable on a machine that has neither and
    cheap enough to call from a diagnostic. ``cuInit`` is idempotent and creates no context,
    so nothing is left behind on the device.

    Every failure -- no driver, no device, an unexpected API -- returns an empty tuple.
    "Which GPUs are there" has an honest answer of "none I can see", and a report is not a
    place to raise.
    """
    library = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        driver = ctypes.CDLL(library)
    except OSError:
        return ()

    try:
        if driver.cuInit(0) != 0:
            return ()
        count = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return ()

        devices: list[CudaDevice] = []
        for index in range(count.value):
            handle = ctypes.c_int()
            if driver.cuDeviceGet(ctypes.byref(handle), index) != 0:
                continue
            name = ctypes.create_string_buffer(256)
            driver.cuDeviceGetName(name, len(name), handle)
            major, minor = ctypes.c_int(), ctypes.c_int()
            # 75 and 76 are CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}. The
            # numbers are part of the driver ABI and have never been renumbered; the
            # named alternative, cuDeviceComputeCapability, is deprecated.
            driver.cuDeviceGetAttribute(ctypes.byref(major), 75, handle)
            driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, handle)
            memory = ctypes.c_size_t()
            total_mem = getattr(driver, "cuDeviceTotalMem_v2", None) or driver.cuDeviceTotalMem
            total_mem(ctypes.byref(memory), handle)
            devices.append(
                CudaDevice(
                    index=index,
                    name=name.value.decode("utf-8", "replace"),
                    compute_capability=(major.value, minor.value),
                    total_memory_bytes=int(memory.value),
                )
            )
        return tuple(devices)
    except (AttributeError, OSError):  # pragma: no cover -- not the CUDA driver after all
        return ()


@cache
def gtsam_backend() -> GtsamBackend:
    """Load GTSAM if it is there, and describe it. Never raises; cached per process.

    This is the function for a diagnostic or a ``pytest.mark.skipif``. Use
    [`import_gtsam`][ocbf.backends.import_gtsam] when the module itself is wanted and its absence is an error.
    """
    searched = enable_cuda_library_search()
    try:
        import gtsam
    except ImportError as exc:
        return GtsamBackend(
            available=False, searched=searched, reason=_explain_import_failure(exc, searched)
        )

    try:
        from importlib.metadata import version

        installed = version("gtsam")
    except Exception:  # pragma: no cover -- a source checkout with no dist metadata
        installed = getattr(gtsam, "__version__", None)

    return GtsamBackend(
        available=True,
        version=installed,
        cuda_built=hasattr(gtsam, "cuda"),
        devices=cuda_devices(),
        searched=searched,
        module=gtsam,
    )


def import_gtsam() -> ModuleType:
    """Import GTSAM, having first made its CUDA dependencies resolvable.

    Raises:
        BackendUnavailable: if GTSAM is missing or will not load. The message names the
            directories that were searched, which is the part a traceback omits.
    """
    backend = gtsam_backend()
    if backend.module is None:
        raise BackendUnavailable(backend.reason)
    return backend.module


def _explain_import_failure(exc: ImportError, searched: tuple[Path, ...]) -> str:
    """Turn an opaque loader failure into something a reader can act on."""
    if "No module named" in str(exc):
        return f"gtsam is not installed ({exc})"
    where = ", ".join(str(p) for p in searched) or "nowhere -- no CUDA toolkit was found"
    return (
        f"gtsam is installed but its native library would not load ({exc}). "
        f"Searched for the CUDA runtime in: {where}. "
        f"Set {CUDA_BIN_ENV} if it lives somewhere else."
    )


def report() -> str:
    """A formatted account of the optional backends, for a log or a bug report."""
    backend = gtsam_backend()
    lines = [
        "Optional backends",
        "-" * 72,
        f"  gtsam       {backend.summary()}",
        f"  CUDA search {', '.join(str(p) for p in backend.searched) or '(nothing found)'}",
    ]
    devices = cuda_devices()
    if devices:
        lines.extend(f"  cuda:{d.index}      {d.summary()}" for d in devices)
    else:
        lines.append("  cuda        no device visible to the driver")
    return "\n".join(lines)
