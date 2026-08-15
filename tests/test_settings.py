from realsense_nav import settings


def test_local_python_config_is_used(monkeypatch) -> None:
    monkeypatch.delenv("TEST_REALSENSE_SETTING", raising=False)
    module = settings._local_config()
    assert module is not None
    monkeypatch.setattr(module, "TEST_REALSENSE_SETTING", "from-python-file", raising=False)
    assert settings.get_setting("TEST_REALSENSE_SETTING") == "from-python-file"


def test_environment_has_priority_over_python_config(monkeypatch) -> None:
    module = settings._local_config()
    assert module is not None
    monkeypatch.setattr(module, "TEST_REALSENSE_PRIORITY", "from-python-file", raising=False)
    monkeypatch.setenv("TEST_REALSENSE_PRIORITY", "from-environment")
    assert settings.get_setting("TEST_REALSENSE_PRIORITY") == "from-environment"

