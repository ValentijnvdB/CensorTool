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


def test_python_backend_defaults() -> None:
    """The defaults in docs/censor_config.md (python backend only; the C++ backend has its own)."""
    from core.censor_config import CsBlur, CsPixel, censor_config_from_dict

    cfg = censor_config_from_dict({
        "features_to_censor": ["FACE_MALE", "FACE_FEMALE", "EYES", "FEET", "ARMPITS"],
        "default_censor_config": {"censor_style": {"type": "blur"}},
        "feature_overrides": {
            "FACE_FEMALE": {"censor_style": {"type": "pixel"}},
            "EYES": {"censor_style": {"type": "no_such_style"}},
            "FEET": {"censor_style": {}},
            "ARMPITS": {"width_area_safety": 0.3},
        },
    })
    boxes = cfg.features_to_censor
    assert boxes["FACE_MALE"].style == CsBlur(31)
    assert boxes["FACE_FEMALE"].style == CsPixel(20)
    assert boxes["EYES"].style == CsBlur(31)      # unknown type
    assert boxes["FEET"].style == CsBlur(31)      # missing type
    assert cfg.inverse_censor_style == CsBlur(31)  # missing inverse_censor_style
    assert (boxes["FACE_MALE"].width_area_safety, boxes["FACE_MALE"].height_area_safety) == (0.1, 0.1)
    assert (boxes["ARMPITS"].width_area_safety, boxes["ARMPITS"].height_area_safety) == (0.3, 0.1)
