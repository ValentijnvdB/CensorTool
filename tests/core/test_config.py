"""Censor config parsing."""
from __future__ import annotations

from scenarios import SCENARIOS


def test_unknown_keys_warn_with_suggestion(cc, capfd) -> None:
    cfg = {
        "features_to_censor": ["FACE_MALE"],
        "default_censor_config": {"censor_style": {"type": "bar", "color": [0, 0, 0]}, "min_prob": 0.25},
        "inverse_censor_style": {"type": "blur", "blur_strength": 51},
        "feature_overrides": {"FACE_MALE": {"shap": "circle"}},
    }
    cc.Job(config=cfg)
    err = capfd.readouterr().err
    assert "'min_prob' in default_censor_config" in err and "did you mean 'min_probability'" in err
    assert "'blur_strength' in inverse_censor_style" in err and "did you mean 'strength'" in err
    assert "'shap' in feature_overrides.FACE_MALE" in err and "did you mean 'shape'" in err


def test_known_keys_do_not_warn(cc, capfd) -> None:
    for _, cfg in SCENARIOS.values():
        cc.Job(config=cfg)
    assert "Unknown key" not in capfd.readouterr().err


def test_load_censor_config_returns_usable_dict(cc, tmp_path) -> None:
    path = tmp_path / "cfg.yml"
    path.write_text(
        "features_to_censor: [FACE_MALE]\n"
        "default_censor_config:\n"
        "  censor_style: {type: pixel, factor: 10}\n"
        "  min_probability: 0.4\n"
    )
    cfg = cc.load_censor_config(str(path))
    assert cfg["features_to_censor"] == ["FACE_MALE"]
    assert cfg["default_censor_config"]["min_probability"] == 0.4
    cc.Job(config=cfg)  # accepted as-is
