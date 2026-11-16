"""Static Plotly export with one narrowly scoped Kaleido bootstrap retry.

Kaleido's embedded Chrome context can occasionally report a missing
kaleido_scopes JavaScript global during initialization. Retrying *once*
is appropriate for that specific transient startup error; every other
failure, and a repeated startup failure, must propagate unchanged.
"""


def write_image_with_bootstrap_retry(figure, path, **kwargs):
    """Write the image; retry only an initial missing Kaleido JS scope."""
    try:
        return figure.write_image(str(path), **kwargs)
    except Exception as exc:
        if "kaleido_scopes is not defined" not in str(exc):
            raise
    # Do not silently downgrade to HTML or claim PNG success on failure.
    # If the second attempt fails, preserve that original exception.
    return figure.write_image(str(path), **kwargs)
