from realsense_nav import settings


def test_local_python_config_is_used(monkeypatch) -> None:
    module = settings._local_config()
    assert module is not None
    monkeypatch.setattr(module, "TEST_REALSENSE_SETTING", "from-python-file", raising=False)
    assert settings.get_setting("TEST_REALSENSE_SETTING") == "from-python-file"


def test_environment_is_not_used(monkeypatch) -> None:
    module = settings._local_config()
    assert module is not None
    monkeypatch.setattr(module, "TEST_REALSENSE_PRIORITY", "from-python-file", raising=False)
    monkeypatch.setenv("TEST_REALSENSE_PRIORITY", "from-environment")
    assert settings.get_setting("TEST_REALSENSE_PRIORITY") == "from-python-file"


def test_default_is_used_when_local_config_has_no_value(monkeypatch) -> None:
    module = settings._local_config()
    assert module is not None
    monkeypatch.delattr(module, "TEST_REALSENSE_DEFAULT", raising=False)
    monkeypatch.setenv("TEST_REALSENSE_DEFAULT", "from-environment")
    assert settings.get_setting("TEST_REALSENSE_DEFAULT", "fallback") == "fallback"
