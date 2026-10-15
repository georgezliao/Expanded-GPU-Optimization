from fused_hoad._extension import load_extension


def test_unbuilt_extension_has_actionable_error() -> None:
    try:
        load_extension()
    except RuntimeError as error:
        assert "pip install -e . --no-build-isolation" in str(error)

