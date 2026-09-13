"""Located failures shared by evidence, model, inference and query boundaries."""


class OCBFError(ValueError):
    """Invalid scientific input or requested operation, with a stable location."""

    def __init__(self, message: str, *, key: str | None = None):
        self.key = key
        self.execution = None
        super().__init__(f"{key}: {message}" if key else message)


class ValidationError(OCBFError):
    """A contract is invalid."""


class EvidenceConflict(OCBFError):
    """Revision actions cannot be resolved unambiguously."""


class CapabilityError(OCBFError):
    """The requested computation is unsupported."""


class BudgetExceeded(CapabilityError):
    """A symbolic resource estimate exceeds an explicit budget."""


class IncompatibleModel(OCBFError):
    """The declared model has zero probability mass."""


class NumericalFailure(OCBFError):
    """A numerical calculation could not establish a valid result."""


class ExecutionStopped(OCBFError):
    """Runtime stop, deliberately outside CapabilityError to prevent solver fallback."""

    status = "failed"


class ExecutionFailure(OCBFError):
    """Unexpected implementation failure; the original exception is retained as its cause."""


class ExecutionCancelled(ExecutionStopped):
    status = "cancelled"


class ResourceExhausted(ExecutionStopped):
    status = "resource-exhausted"
