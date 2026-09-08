"""Compatibility shims for third-party dependency API drift.

Kept dependency-light on purpose: this module must not import anything from the
rest of ``optstop`` (so any module can import it without circular-import risk)
and it defers the ``arviz`` import into the function body (mirroring the
lazy-import style used elsewhere in the package).
"""


def check_arviz_version():
    """Fail fast with a clear message if an unsupported arviz major is installed.

    arviz 1.x changed the ``hdi`` return structure; optstop supports the 0.x line
    only (see requirements-lock.txt). The pins should prevent this, but a forced
    install can still bring arviz>=1.0 into the environment.
    """
    from importlib.metadata import version

    ver = version("arviz")
    if int(ver.split(".")[0]) >= 1:
        raise ImportError(
            f"optstop requires arviz<1.0 (found {ver}). arviz 1.x changed the "
            "hdi() return structure and is not supported. Install the validated "
            "stack with: pip install -r requirements-lock.txt"
        )


def hdi(ary, prob, **kwargs):
    """Version-tolerant wrapper around :func:`arviz.hdi`.

    ``arviz-stats`` 1.0 renamed the ``hdi_prob`` keyword to ``prob``; passing the
    old name on that stack raises ``TypeError`` ("hdi got an unexpected keyword
    argument: 'hdi_prob'"). Older/validated arviz (the pinned 0.x line, e.g.
    0.23.4) only accepts ``hdi_prob``.

    We try the validated-stack keyword first so the common, tested path never
    raises, then fall back to the arviz-1.x keyword. If the fallback also raises
    ``TypeError`` the error is not about the keyword rename (e.g. a genuinely
    malformed ``ary``), so we re-raise the original, more informative exception.

    Note: this only shims the ``hdi`` keyword. It does not claim full arviz-1.x
    support - the pinned stack in ``requirements-lock.txt`` remains the validated
    configuration (see issue #4).
    """
    import arviz as az

    try:
        return az.hdi(ary, hdi_prob=prob, **kwargs)
    except TypeError as first_err:
        try:
            return az.hdi(ary, prob=prob, **kwargs)
        except TypeError:
            raise first_err
