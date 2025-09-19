"""
Adaptibot package initializer.

This file makes the `adaptibot` folder a Python package.
It also provides:
- A version string (`__version__`) so tools and reports know the package version.
- An `__all__` list to define what submodules are meant to be public.
"""

# Package version
__version__ = "1.0.0"

# Explicitly exported submodules
__all__ = [
    "app",        # Command-line entry and main loop
    "controller", # Core environment logic and rescue controller
    "config",     # Dataclasses and enums for configuration
    "policy",     # PPO model loader and policy wrapper
    "runtime",    # Helpers for runtime setup (seeds, devices, etc.)
    "terrain",    # Environment terrain setup and modifiers
]
