"""Minimal pytest.raises polyfill for environments without pytest installed."""
import contextlib

class _RaisesContext:
    def __init__(self, expected_exception):
        self.expected_exception = expected_exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.expected_exception.__name__} but no exception was raised")
        if not issubclass(exc_type, self.expected_exception):
            return False  # Re-raise unexpected exception
        return True  # Suppress expected exception

def raises(expected_exception):
    return _RaisesContext(expected_exception)
