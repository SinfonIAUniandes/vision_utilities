import sys

from typing import List

from common.ConsoleFormatter import ConsoleFormatter
from pydantic import BaseModel, Field, ValidationError


class VisionModuleConfiguration(BaseModel):
    with_pepper: bool = Field(default=False)
    start_cameras: bool = Field(default=False)


def parse_config(args: List[str]) -> VisionModuleConfiguration:
    """
    Parses flags avaiable in VisionConfiguration

    Argument format:
    --<name>[=value]
    """
    default_config = VisionModuleConfiguration()
    received_config = default_config.model_dump()
    keys = set(received_config.keys())

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
        ConsoleFormatter.error(
            f"Vision utilities: The provided configuration is not valid. Check for documentation"
        )
        sys.exit(-1)
