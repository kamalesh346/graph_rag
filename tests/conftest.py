import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live Qdrant Cloud credentials (read-only)"
    )
