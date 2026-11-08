"""Profile selection when --profile is not given."""
import pytest
from loguru import logger

import main
import app.config
from app.config import ProfileError, resolve_profile_dir
from utils import cli_select


@pytest.fixture
def profiles(monkeypatch):
    """Set the profiles list_profiles returns and fail if the user would be asked."""
    def set_profiles(names, interactive=True):
        monkeypatch.setattr(app.config, 'list_profiles', lambda: list(names))
        monkeypatch.setattr(cli_select, 'is_interactive', lambda: interactive)
        monkeypatch.setattr(cli_select, 'select', lambda *a, **k: pytest.fail("the user should not be asked"))
    return set_profiles


@pytest.fixture
def profiles_dir(tmp_path, monkeypatch):
    """A configs directory with the given profiles."""
    def make(*names):
        for name in names:
            (tmp_path / name).mkdir()
        monkeypatch.setattr(app.config.constants, 'profiles_path', tmp_path)
        return tmp_path
    return make


@pytest.fixture
def warnings():
    messages = []
    handler = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(handler)


def test_only_profile_is_selected_without_asking(profiles):
    profiles(['privacy'])
    assert main.select_profile() == 'privacy'


def test_not_interactive_leaves_the_choice_to_the_fallback(profiles):
    profiles(['nudity', 'privacy'], interactive=False)
    assert main.select_profile() is None


def test_multiple_profiles_asks_the_user(monkeypatch):
    monkeypatch.setattr(app.config, 'list_profiles', lambda: ['nudity', 'privacy'])
    monkeypatch.setattr(cli_select, 'is_interactive', lambda: True)
    monkeypatch.setattr(cli_select, 'select', lambda prompt, options: options[1])
    assert main.select_profile() == 'privacy'


def test_profiles_are_listed_alphabetically(profiles_dir):
    profiles_dir('privacy', 'Nudity', 'custom')
    assert app.config.list_profiles() == ['custom', 'Nudity', 'privacy']


def test_no_profile_uses_first_alphabetically_with_warning(profiles_dir, warnings):
    root = profiles_dir('privacy', 'nudity')
    assert resolve_profile_dir(None) == root / 'nudity'
    assert any("No profile given, using 'nudity'" in m for m in warnings)


def test_no_profile_with_a_single_profile_does_not_warn(profiles_dir, warnings):
    root = profiles_dir('privacy')
    assert resolve_profile_dir(None) == root / 'privacy'
    assert warnings == []


def test_no_profiles_at_all_is_an_error(profiles_dir):
    profiles_dir()
    with pytest.raises(ProfileError, match="No profiles found"):
        resolve_profile_dir(None)


def test_profile_without_config_yml_is_an_error(profiles_dir):
    root = profiles_dir('custom')
    (root / 'custom' / 'censor_config.yml').write_text('features_to_censor: []')
    (root / 'custom' / 'pipeline.yml').write_text('models: []')
    with pytest.raises(ProfileError, match="has no config.yml"):
        app.config.resolve_profile_files('custom')
