"""
IBM Enterprise AI/ML Platform — pytest configuration
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "watson: marks tests requiring Watson credentials")
