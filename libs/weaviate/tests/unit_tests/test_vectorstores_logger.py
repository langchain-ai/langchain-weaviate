"""Test for vectorstores logger initialization (line 31)."""

import importlib
import logging
import sys


def test_vectorstores_logger_init() -> None:
    """Test line 31: logger.setLevel(logging.DEBUG) is called on import.
    
    This test ensures the logger initialization line is executed by
    reloading the module within the test.
    """
    # Remove the module from cache to force fresh import
    if "langchain_weaviate.vectorstores" in sys.modules:
        del sys.modules["langchain_weaviate.vectorstores"]
    
    # Clear any existing handlers/configuration on the logger
    logger = logging.getLogger("langchain_weaviate.vectorstores")
    logger.setLevel(logging.NOTSET)  # Reset to default
    
    # Now import the module - this will execute line 31
    import langchain_weaviate.vectorstores  # noqa: F401
    
    # Verify the logger level was set to DEBUG
    assert logger.level == logging.DEBUG, f"Expected DEBUG ({logging.DEBUG}), got {logger.level}"
