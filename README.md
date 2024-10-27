# Vision Utilities

**Vision Utilities** is a submodule of the **Perception Module** in the **SinfonIA** workspace. It provides essential vision-related functionalities and services for the perception system, including support for QR code recognition.

## Overview

The Vision Utilities submodule offers ROS services tailored for perception tasks, enabling efficient handling of various vision-based operations. The configuration can be tailored based on the environment setup, particularly for robots such as Pepper. The module parses configuration options at runtime, allowing flexible deployment scenarios.

### ROS Services

The following ROS service is provided:

- **`/vision_utilities/recognition/read_qr_srv`**  
  A service for reading and interpreting QR codes within the environment.

## Configuration

Vision Utilities allows configuration via the `VisionModuleConfiguration` class, which specifies essential setup flags. The configuration is handled through command-line arguments passed to the module, enabling or disabling specific functionalities like initializing the robot's camera or enabling features for Pepper robot compatibility.

### Configuration Options

The module configuration includes the following fields:

- **`with_pepper`** (*bool*, default=`False`):  
  Enables compatibility with the Pepper robot, allowing specialized handling for Pepper-specific functionalities.
  
- **`start_cameras`** (*bool*, default=`False`):  
  If `True`, initializes the robot’s cameras during module startup.

### How to Configure

Configuration is passed through command-line arguments in the format `--<name>[=value]`. Below is the structure of how to use this format:

```bash
rosrun vision_utilities vision_utilities.py --with_pepper=True --start_cameras
```

- **Format**: `--<flag_name>[=value]`
  - If a value is not provided, the flag is assumed to be `True`.

### Example

To initialize the module with both `with_pepper` and `start_cameras` enabled:

```bash
rosrun vision_utilities vision_utilities.py --with_pepper --start_cameras
```

In this case, since no explicit value is given, both flags are set to `True`.

**Validation**: Parsed arguments are validated against the `VisionModuleConfiguration` model to ensure correctness. If invalid, the module provides an error and exits.

## Error Handling

Errors during argument parsing will result in descriptive messages, ensuring users are aware of issues like:
- Incorrect argument format.
- Unrecognized or unsupported configuration flags.
- Validation errors on provided configuration values.

## License

This repository is proprietary and is intended solely for use by SInfonIA. Unauthorized use, distribution, or modification is prohibited.
