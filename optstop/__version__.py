"""
optstop version information.

This module provides version information for the optstop package and its components.
"""

# Package version (semantic versioning: MAJOR.MINOR.PATCH)
__version__ = "0.4.0"

# Version as tuple for programmatic comparison
__version_info__ = (0, 4, 0)

# Component versions for transparency
__core_version__ = "0.4.0"  # Core optstop algorithms (posthoc, live, convergence)
__bridge_version__ = "0.4.0"  # inspect_ai bridge (OptimalStoppingManager)

# Minimum compatible inspect_ai version
__min_inspect_ai_version__ = "0.3.0"

# Release metadata
__release_date__ = "2026-03-18"
__status__ = "beta"  # Options: alpha, beta, stable


def get_version():
    """Get the current optstop version string.

    Returns:
        str: Version string (e.g., "0.4.0")
    """
    return __version__


def get_version_info():
    """Get the current optstop version as a tuple.

    Returns:
        tuple: Version tuple (e.g., (0, 4, 0))
    """
    return __version_info__


def check_inspect_ai_compatibility(inspect_ai_version: str) -> bool:
    """Check if a given inspect_ai version is compatible.

    Args:
        inspect_ai_version: Version string to check (e.g., "0.3.0")

    Returns:
        bool: True if compatible, False otherwise

    Example:
        >>> from optstop import __version__
        >>> check_inspect_ai_compatibility("0.3.0")
        True
        >>> check_inspect_ai_compatibility("0.2.0")
        False
    """
    def parse_version(version_str: str) -> tuple:
        """Parse version string to tuple."""
        try:
            parts = version_str.split('.')
            return tuple(int(p) for p in parts[:3])
        except (ValueError, AttributeError):
            return (0, 0, 0)

    current = parse_version(inspect_ai_version)
    minimum = parse_version(__min_inspect_ai_version__)

    return current >= minimum


def print_version_info():
    """Print detailed version information (useful for debugging).

    Example:
        >>> from optstop.__version__ import print_version_info
        >>> print_version_info()
        optstop version: 0.4.0
        Release date: 2026-03-18
        Status: beta
        Core version: 0.4.0
        Bridge version: 0.4.0
        Minimum inspect_ai version: 0.3.0
    """
    print(f"optstop version: {__version__}")
    print(f"Release date: {__release_date__}")
    print(f"Status: {__status__}")
    print(f"Core version: {__core_version__}")
    print(f"Bridge version: {__bridge_version__}")
    print(f"Minimum inspect_ai version: {__min_inspect_ai_version__}")
