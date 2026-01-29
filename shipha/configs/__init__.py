"""
ShiphaEvolve Configuration System
=================================

⚠️  YAML CONFIGURATION IS DEPRECATED  ⚠️

This package uses Hydra for configuration management. YAML files in this
directory are provided ONLY for backward compatibility and WILL BE REMOVED
in a future version.

RECOMMENDED: Use Python dataclasses with Hydra structured configs:

    from shipha.core import EvolutionConfig

    config = EvolutionConfig(
        task_name="my_task",
        max_generations=50,
        llm=LLMConfig(models=["gpt-4", "claude-3-sonnet"]),
        evaluator=EvaluatorConfig(trust_level="SANDBOX"),
    )

See shipha/core/config.py for the full config schema.
"""

import warnings
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent


def load_yaml_config(name: str = "default") -> dict:
    """
    Load a YAML config file. DEPRECATED - use structured configs instead.
    
    Args:
        name: Config name (without .yaml extension)
        
    Returns:
        Config dictionary
        
    Warns:
        DeprecationWarning: YAML configs are deprecated
    """
    warnings.warn(
        "YAML configuration is deprecated. Use Python dataclasses with Hydra structured configs. "
        "See shipha/core/config.py for the recommended approach.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    import yaml
    
    config_path = CONFIGS_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)
