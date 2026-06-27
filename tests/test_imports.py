"""Smoke tests for package importability."""


def test_package_imports_without_syntax_errors():
    """Importing braintop should not raise syntax errors."""
    import braintop

    assert braintop.__version__ == "0.1.0"
