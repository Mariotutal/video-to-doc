import pytest

from vtd import config as config_module


@pytest.fixture(autouse=True)
def no_user_config(tmp_path, monkeypatch):
    """Keep tests independent of whatever config file the developer has."""
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "no-such-config.toml")
