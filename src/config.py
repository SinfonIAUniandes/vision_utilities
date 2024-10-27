import sys

from .common.ConsoleFormatter import ConsoleFormatter
from pydantic import BaseModel, Field, ValidationError


class VisionConfiguration(BaseModel):
    with_pepper: bool = Field(default=False)


def parse_config(args: list[str]) -> VisionConfiguration:
    """
    Parses flags avaiable in VisionConfiguration

    Argument format:
    --name[=value]
    """
    default_config = VisionConfiguration()
    received_config = default_config.model_dump()
    keys = set(received_config.keys())

    for i in range(len(args)):
        argument = args[i]
        if len(argument.split("--")) < 2:
            ConsoleFormatter.error(f"Vision utilities: Argument in position `{i}` has not the appropiate format: --(name)[=value]")
            sys.exit(-1)
        parts = argument.split("=", 1)

        name = parts[0][2:]

        if name not in keys:
            ConsoleFormatter.error(f"Vision utilities: Argument `--{name}` is not valid. Check for documentation")
            sys.exit(-1)

        if len(parts) > 1: # Value has been provided
            received_config[name] = parts[1]
        else: # Assuming true as the value was not provided
            received_config[name] = True

    try:
        return VisionConfiguration.model_validate(received_config)
    except ValidationError as err:
        ConsoleFormatter.error(f"Vision utilities: The provided configuration is not valid. Check for documentation")
        sys.exit(-1)

