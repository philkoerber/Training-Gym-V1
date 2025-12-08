"""Backward compatibility wrapper for Lightning App.

This script imports from the new modular structure and maintains
backward compatibility with existing workflows.
"""

from src.training.train_app import app

# Export the app for Lightning CLI
__all__ = ["app"]
