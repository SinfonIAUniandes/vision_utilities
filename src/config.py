import sys
from pathlib import Path
from typing import List

from common.ConsoleFormatter import ConsoleFormatter
from pydantic import BaseModel, Field, ValidationError

import yaml


class VisionModuleConfiguration(BaseModel):
    with_pepper: bool = Field(default=False)
    start_cameras: bool = Field(default=False)
    # VLM configuration (loaded from config.yaml under `vlm:`)
    llm_mode: str = Field(default="ollama")
    vlm_model: str = Field(default="gemma3_12b")
    vlm_max_tokens: int = Field(default=500)


def parse_config(args: List[str]) -> VisionModuleConfiguration:
    """
    Parses flags avaiable in VisionConfiguration

    Argument format:
    --<name>[=value]
    """
    default_config = VisionModuleConfiguration()
    received_config = default_config.model_dump()
    keys = set(received_config.keys())

    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists() and yaml is not None:
        with config_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        vlm_cfg = data.get("vlm") or {}
        # Map YAML keys to our model's keys
        if isinstance(vlm_cfg, dict):
            if "llm_mode" in vlm_cfg:
                received_config["llm_mode"] = vlm_cfg.get("llm_mode")
            if "model" in vlm_cfg:
                received_config["vlm_model"] = vlm_cfg.get("model")
            if "max_tokens" in vlm_cfg:
                received_config["vlm_max_tokens"] = vlm_cfg.get("max_tokens")
    elif config_path.exists() and yaml is None:
        print(ConsoleFormatter.warning("Vision utilities: PyYAML not installed; skipping config.yaml."))

    for i in range(len(args)):
        argument = args[i]
        if len(argument.split("--")) < 2:
            print(
                ConsoleFormatter.error(
                    f"Vision utilities: Argument in position `{i}` has not the appropiate format: --<name>[=value]"
                )
            )
            sys.exit(-1)
        parts = argument.split("=", 1)

        name = parts[0][2:]

        if name not in keys:
            print(
                ConsoleFormatter.error(
                    f"Vision utilities: Argument `--{name}` is not valid. Check for documentation"
                )
            )
            sys.exit(-1)

        if len(parts) > 1:  # Value has been provided
            received_config[name] = parts[1]
        else:  # Assuming true as the value was not provided
            received_config[name] = True

    try:
        return VisionModuleConfiguration.model_validate(received_config)
    except ValidationError:
        print(
            ConsoleFormatter.error(
                f"Vision utilities: The provided configuration is not valid. Check for documentation"
            )
        )
        sys.exit(-1)
