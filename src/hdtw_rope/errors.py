"""Domain-specific failures that enforce fail-closed behavior."""


class HDTWRoPEError(RuntimeError):
    """Base exception for the package."""


class InvalidShapeError(HDTWRoPEError, ValueError):
    """Raised when a public tensor contract is violated."""


class NoValidAlignmentPath(HDTWRoPEError):
    """Raised when the configured band or masks admit no DTW path."""


class InsufficientAlignmentMass(HDTWRoPEError):
    """Raised when an alignment row or column cannot define a clock."""


class InvalidClockError(HDTWRoPEError):
    """Raised for nonfinite, malformed, or schema-incompatible clocks."""


class InvalidHierarchyError(HDTWRoPEError):
    """Raised when hierarchy intervals or parent relationships are invalid."""
