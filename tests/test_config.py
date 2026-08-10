from pathlib import Path

from scout_embedding.config import ExperimentConfig


def test_scout_config_is_explicit_and_portable():
    config = ExperimentConfig.from_yaml(Path("configs/scout.yaml"))
    assert config.scout.selection_ratio == 1.75
    assert config.scout.prompt_attention_scope == "local"
    assert config.scout.compression_prompt == '\nBrief summary:"'
    assert not config.model.name_or_path.startswith("/")
