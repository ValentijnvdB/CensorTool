"""Censor configs: sticker paths are limited to data/stickers, and the ComfyUI settings come from config.yml."""
import pytest
import yaml

import app.config
from app.config import (CONFIG, CensorConfigError, apply_comfy_settings, check_client_censor_config,
                        read_censor_config_from_file, restrict_sticker_paths)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A working directory with data/stickers/plants, and a folder next to it that must not be reachable."""
    root = tmp_path / 'instance'
    stickers = root / 'data' / 'stickers'
    (stickers / 'plants').mkdir(parents=True)
    (tmp_path / 'secret').mkdir()
    monkeypatch.setattr(app.config.constants, 'instance_dir', root)
    monkeypatch.setattr(app.config.constants, 'stickers_root_path', stickers)
    return root


def config(stickers_dir=None, stickers=None, override_stickers=None, key='stickers'):
    c = {'features_to_censor': ['face'],
         'default_censor_config': {'censor_style': {'type': 'bar'}}}
    if stickers_dir is not None:
        c['stickers_dir'] = stickers_dir
    if stickers is not None:
        c['default_censor_config']['overlay'] = {'type': 'sticker', key: stickers}
    if override_stickers is not None:
        c['feature_overrides'] = {'face': {'overlay': {'type': 'sticker', key: override_stickers}}}
    return c


@pytest.mark.parametrize('stickers_dir', ['./data/stickers', 'data/stickers', 'data/stickers/plants',
                                          'data/stickers/not_created_yet', 'data/stickers/plants/..'])
def test_allowed_stickers_dir_is_made_absolute(project, stickers_dir):
    result = restrict_sticker_paths(config(stickers_dir))
    expected = (project / stickers_dir).resolve()
    assert result['stickers_dir'] == str(expected)


def test_absolute_stickers_dir_inside_stickers_folder_is_allowed(project):
    path = str(project / 'data' / 'stickers' / 'plants')
    assert restrict_sticker_paths(config(path))['stickers_dir'] == path


def test_missing_stickers_dir_defaults_to_stickers_folder(project):
    result = restrict_sticker_paths(config())
    assert result['stickers_dir'] == str((project / 'data' / 'stickers').resolve())


@pytest.mark.parametrize('stickers_dir', ['data', 'data/stickers/..', '../secret', '/etc',
                                          'data/stickers/../../../secret', 'data/stickers_other'])
def test_stickers_dir_outside_stickers_folder_is_rejected(project, stickers_dir):
    with pytest.raises(CensorConfigError, match='stickers_dir'):
        restrict_sticker_paths(config(stickers_dir))


def test_stickers_dir_symlink_leaving_stickers_folder_is_rejected(project, tmp_path):
    (project / 'data' / 'stickers' / 'link').symlink_to(tmp_path / 'secret')
    with pytest.raises(CensorConfigError):
        restrict_sticker_paths(config('data/stickers/link'))


@pytest.mark.parametrize('stickers', [[], ['plants'], ['plants', 'misc'], ['plants/flowers'], ['star.png'],
                                      ['plants', 'plants/flowers/rose.png']])
def test_stickers_inside_stickers_dir_are_allowed(project, stickers):
    restrict_sticker_paths(config(stickers=stickers, override_stickers=stickers))


@pytest.mark.parametrize('sticker', ['..', '../../secret', '/etc', 'plants/../../..', '../x.png', '/etc/passwd'])
@pytest.mark.parametrize('key', ['stickers', 'categories'])
def test_sticker_leaving_stickers_folder_is_rejected(project, sticker, key):
    with pytest.raises(CensorConfigError, match='default_censor_config'):
        restrict_sticker_paths(config(stickers=[sticker], key=key))
    with pytest.raises(CensorConfigError, match='feature_overrides.face'):
        restrict_sticker_paths(config(override_stickers=['plants', sticker], key=key))


def test_sticker_may_not_leave_a_stickers_subfolder_into_its_parent(project):
    # '..' from data/stickers/plants is still inside data/stickers, so it is allowed
    restrict_sticker_paths(config('data/stickers/plants', stickers=['..']))


def test_sticker_symlink_leaving_stickers_folder_is_rejected(project, tmp_path):
    (tmp_path / 'secret' / 'x.png').write_bytes(b'')
    (project / 'data' / 'stickers' / 'x.png').symlink_to(tmp_path / 'secret' / 'x.png')
    with pytest.raises(CensorConfigError):
        restrict_sticker_paths(config(stickers=['x.png']))


def test_non_mapping_and_odd_values_are_left_alone(project):
    assert restrict_sticker_paths(None) is None
    assert restrict_sticker_paths(['x']) == ['x']
    c = config()
    c['feature_overrides'] = {'face': None, 'eyes': {'overlay': None}}
    c['default_censor_config']['overlay'] = {'type': 'text', 'values': ['x']}
    restrict_sticker_paths(c)
    c['default_censor_config']['overlay'] = {'type': 'sticker', 'stickers': 'x', 'categories': None}
    restrict_sticker_paths(c)


def test_censor_config_file_is_checked(project, tmp_path):
    path = tmp_path / 'censor_config.yml'
    path.write_text(yaml.safe_dump(config('../secret')))
    with pytest.raises(CensorConfigError, match='censor_config.yml'):
        read_censor_config_from_file(path)

    path.write_text(yaml.safe_dump(config('data/stickers/plants')))
    assert read_censor_config_from_file(path)['stickers_dir'] == str(project / 'data' / 'stickers' / 'plants')


# ---------------------------------------------------------------------------
# ComfyUI settings
# ---------------------------------------------------------------------------

@pytest.fixture
def comfy(monkeypatch):
    monkeypatch.setattr(CONFIG, 'comfy_host', 'comfy.local')
    monkeypatch.setattr(CONFIG, 'comfy_port', 8188)
    monkeypatch.setattr(CONFIG, 'comfy_workflow', 'data/workflow.json')
    return {'comfy_host': 'comfy.local', 'comfy_port': 8188, 'comfy_workflow': 'data/workflow.json'}


def ai_remove_config(**style):
    return {'features_to_censor': ['face', 'eyes'],
            'default_censor_config': {'censor_style': {'type': 'ai_remove', **style}},
            'feature_overrides': {'face': {'censor_style': {'type': 'ai_remove', **style}},
                                  'eyes': {'censor_style': {'type': 'bar'}}, 'other': None}}


def test_missing_comfy_settings_are_filled_in(comfy):
    c = apply_comfy_settings(ai_remove_config(), override=False)
    assert c['default_censor_config']['censor_style'] == {'type': 'ai_remove', **comfy}
    assert c['feature_overrides']['face']['censor_style'] == {'type': 'ai_remove', **comfy}
    assert c['feature_overrides']['eyes']['censor_style'] == {'type': 'bar'}


def test_comfy_settings_in_the_file_are_kept(comfy):
    c = apply_comfy_settings(ai_remove_config(comfy_host='other.local'), override=False)
    assert c['default_censor_config']['censor_style']['comfy_host'] == 'other.local'


def test_client_comfy_settings_are_replaced(project, comfy):
    c = check_client_censor_config(ai_remove_config(comfy_host='evil.example', comfy_port=1,
                                                    comfy_workflow='/home/me/.ssh/id_rsa'))
    assert c['default_censor_config']['censor_style'] == {'type': 'ai_remove', **comfy}
    assert c['feature_overrides']['face']['censor_style'] == {'type': 'ai_remove', **comfy}


def test_client_sticker_paths_are_checked(project, comfy):
    with pytest.raises(CensorConfigError):
        check_client_censor_config(config('/etc'))


# ---------------------------------------------------------------------------
# Picking a sticker: categories + individual stickers form one pool
# ---------------------------------------------------------------------------

@pytest.fixture
def sticker_folder(tmp_path):
    from core import boxes
    boxes._sticker_cache.clear()
    base = tmp_path / 'stickers'
    for f in ('plants/fern.png', 'plants/flowers/rose.png', 'misc/star.png', 'misc/notes.txt', 'cat.jpg'):
        (base / f).parent.mkdir(parents=True, exist_ok=True)
        (base / f).write_bytes(b'')
    yield base
    boxes._sticker_cache.clear()


def names(base, files):
    return sorted(f.relative_to(base.absolute()).as_posix() for f in files)


def test_no_stickers_picks_from_all(sticker_folder):
    from core.boxes import _sticker_files
    assert names(sticker_folder, _sticker_files(sticker_folder, [])) == \
        ['cat.jpg', 'misc/star.png', 'plants/fern.png', 'plants/flowers/rose.png']


def test_folders_and_single_stickers_form_one_pool(sticker_folder):
    from core.boxes import _sticker_files
    files = _sticker_files(sticker_folder, ['plants', 'cat.jpg', 'plants/fern.png'])
    assert names(sticker_folder, files) == ['cat.jpg', 'plants/fern.png', 'plants/flowers/rose.png']


def test_missing_and_non_image_entries_are_skipped(sticker_folder):
    from core.boxes import _sticker_files
    files = _sticker_files(sticker_folder, ['misc/star.png', 'misc/notes.txt', 'missing.png', 'missing_dir'])
    assert names(sticker_folder, files) == ['misc/star.png']


def test_overlay_stickers_are_parsed():
    from core.censor_config import _parse_overlay, OvSticker
    ov = _parse_overlay({'type': 'sticker', 'stickers': ['plants', 'cat.jpg']}, 'test')
    assert ov == OvSticker(probability=1.0, stickers=('plants', 'cat.jpg'))
    assert _parse_overlay({'type': 'sticker'}, 'test').stickers == ()


def test_deprecated_categories_are_added_to_stickers():
    from core.censor_config import _parse_overlay
    ov = _parse_overlay({'type': 'sticker', 'stickers': ['cat.jpg', 'misc'], 'categories': ['plants', 'misc']}, 'test')
    assert ov.stickers == ('cat.jpg', 'misc', 'plants')
