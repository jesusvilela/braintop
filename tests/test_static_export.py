"""Release image export retries only known transient Kaleido JS initialization."""
import pytest

from braintop.utils.static_export import write_image_with_bootstrap_retry


class FakeFigure:
    def __init__(self, failure=None, persistent=False):
        self.failure = failure
        self.persistent = persistent
        self.calls = []

    def write_image(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.failure is not None and (self.persistent or len(self.calls) == 1):
            raise RuntimeError(self.failure)
        return "generated"


def test_single_transient_scope_bootstrap_is_retried():
    figure = FakeFigure("ReferenceError: kaleido_scopes is not defined")
    assert write_image_with_bootstrap_retry(
        figure, "render.png", width=1600, scale=2
    ) == "generated"
    assert figure.calls == [
        ("render.png", {"width": 1600, "scale": 2}),
        ("render.png", {"width": 1600, "scale": 2}),
    ]


def test_non_bootstrap_export_errors_are_not_hidden():
    figure = FakeFigure("Chrome executable not found")
    with pytest.raises(RuntimeError, match="Chrome executable"):
        write_image_with_bootstrap_retry(figure, "render.png")
    assert len(figure.calls) == 1


def test_repeated_bootstrap_failure_propagates():
    figure = FakeFigure(
        "ReferenceError: kaleido_scopes is not defined", persistent=True
    )
    with pytest.raises(RuntimeError, match="kaleido_scopes"):
        write_image_with_bootstrap_retry(figure, "render.png")
    assert len(figure.calls) == 2


def test_successful_export_does_not_retry():
    figure = FakeFigure()
    assert write_image_with_bootstrap_retry(figure, "render.png") == "generated"
    assert len(figure.calls) == 1
