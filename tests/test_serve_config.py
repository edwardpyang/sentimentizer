"""Validate the configuration boundary before constructing the service."""

import pytest

from sentimentizer.serve.config import load_serve_config


@pytest.mark.parametrize("content", ["[]", "false", "42", "text"])
def test_configuration_requires_a_yaml_mapping(tmp_path, content):
    path = tmp_path / "service.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="mapping"):
        load_serve_config(path)


@pytest.mark.parametrize("content", ["serve_host: []", "serve_port: []", "unknown_option: 1"])
def test_configuration_rejects_invalid_field_types_and_unknown_options(tmp_path, content):
    path = tmp_path / "service.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_serve_config(path)


def test_environment_overrides_yaml_and_preserves_typed_values(tmp_path, monkeypatch):
    path = tmp_path / "service.yaml"
    path.write_text("serve_port: 8080\ncors_origins: [https://example.com]\n")
    monkeypatch.setenv("SENTIMENTIZER_SERVE_PORT", "9000")
    monkeypatch.setenv("SENTIMENTIZER_EMBEDDINGS_ENABLED", "false")
    config = load_serve_config(path)
    assert config.serve_port == 9000
    assert config.cors_origins == ["https://example.com"]
    assert config.embeddings_enabled is False
