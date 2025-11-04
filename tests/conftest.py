"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio backend for async tests."""
    return "asyncio"

