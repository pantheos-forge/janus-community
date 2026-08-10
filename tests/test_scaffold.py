def test_janus_package_imports():
    import janus
    assert janus.__version__ == "0.1.0"


def test_core_subpackages_import():
    import janus.core
    import janus.core.tools  # noqa: F401
