#! /usr/bin/env python

"""
Convert a ULog file into MCAP file using Foxglove schemas
"""

from collections import defaultdict
import argparse
import inspect
import itertools
import json
import typing

import numpy as np
from foxglove.schemas import FrameTransform, LocationFix, Log, PosesInFrame
from google.protobuf.json_format import MessageToDict
from mcap.writer import Writer, CompressionType  # pylint: disable=import-error

from .core import ULog

#pylint: disable=too-many-locals, invalid-name, too-many-branches, too-many-statements, too-many-return-statements

# Path segment length in meters for flight path tracking
PATH_SEGMENT_LENGTH = 0.1

# MCAP timestamps are uint64 (nanoseconds); clamp to valid range
MCAP_TIMESTAMP_MAX = (1 << 64) - 1

FOXGLOVE_SCHEMA_CLASSES = {
    "LocationFix": LocationFix,
    "FrameTransform": FrameTransform,
    "PosesInFrame": PosesInFrame,
    "Log": Log,
}


def _get_primitive_type(ulog_type: str) -> typing.Optional[str]:
    """Map ULog type to JSON schema primitive type"""
    type_mapping = {
        "int8_t": "integer",
        "uint8_t": "integer",
        "int16_t": "integer",
        "uint16_t": "integer",
        "int32_t": "integer",
        "uint32_t": "integer",
        "int64_t": "integer",
        "uint64_t": "integer",
        "float": "number",
        "double": "number",
        "bool": "boolean",
        "char": "string",
    }
    return type_mapping.get(ulog_type)


def get_absolute_timestamp_reference(
    ulog: ULog,
) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
    """
    Extract absolute UTC timestamp reference from GPS or time_ref_utc.

    Returns:
        Tuple of (reference_time_us, first_gps_timestamp_us) or (None, None) if not available
    """
    # Check for time_ref_utc in info messages (UTC timestamp in microseconds)
    time_ref_utc = ulog.msg_info_dict.get('time_ref_utc', 0)

    if time_ref_utc and time_ref_utc > 0:
        # time_ref_utc is the UTC timestamp when logging started
        return time_ref_utc, ulog.start_timestamp

    # Try to get GPS data for absolute time reference
    try:
        gps_data = ulog.get_dataset('vehicle_gps_position')
        if gps_data and len(gps_data.data['timestamp']) > 0:
            if 'time_utc_usec' in gps_data.data:
                gps_time_utc_usec = gps_data.data['time_utc_usec'][0]  # First GPS timestamp
                first_gps_timestamp_us = gps_data.data['timestamp'][0]  # Relative timestamp
                if gps_time_utc_usec > 0:
                    return gps_time_utc_usec, first_gps_timestamp_us
    except (KeyError, IndexError, TypeError, ValueError):
        pass  # No GPS data available

    return None, None


def convert_timestamp(relative_timestamp_us: int, reference_time_us: int,
                     first_relative_timestamp_us: int) -> int:
    """
    Convert relative ULog timestamp to absolute nanoseconds for MCAP.

    Args:
        relative_timestamp_us: Relative timestamp from ULog (microseconds)
        reference_time_us: Absolute UTC reference time (microseconds)
        first_relative_timestamp_us: First relative timestamp (for offset calculation)

    Returns:
        Absolute timestamp in nanoseconds
    """
    # Convert to Python ints to avoid numpy uint64 overflow issues
    rel_ts = int(relative_timestamp_us)
    first_ts = int(first_relative_timestamp_us)
    ref_ts = int(reference_time_us)

    # Calculate offset from first timestamp
    offset_us = rel_ts - first_ts

    # Add offset to reference time and convert to nanoseconds
    absolute_time_ns = (ref_ts + offset_us) * 1000
    return absolute_time_ns


def relative_timestamp_to_ns(
    relative_timestamp_us: int,
    use_absolute_time: bool,
    reference_time_us: typing.Optional[int],
    first_relative_timestamp_us: typing.Optional[int],
) -> int:
    """Convert a ULog relative timestamp to MCAP nanoseconds and clamp to uint64."""
    if use_absolute_time:
        absolute_timestamp_ns = convert_timestamp(
            relative_timestamp_us,
            reference_time_us,
            first_relative_timestamp_us,
        )
    else:
        absolute_timestamp_ns = int(relative_timestamp_us) * 1000
    return max(0, min(MCAP_TIMESTAMP_MAX, int(absolute_timestamp_ns)))


def ned_to_enu_position(x: float, y: float, z: float) -> typing.Dict[str, float]:
    """
    Convert position from NED to ENU frame.
    NED: X=North, Y=East, Z=Down
    ENU: X=East, Y=North, Z=Up
    """
    return {
        "x": y,   # East = East
        "y": x,   # North = North
        "z": -z,  # Up = -Down
    }


def ned_quaternion_to_enu(q: typing.List[float]) -> typing.Dict[str, float]:
    """
    Convert quaternion from NED to ENU frame.
    The quaternion represents rotation from body (FRD) to earth (NED).
    We convert it to body (FLU) to earth (ENU).
    """
    w, x, y, z = q
    return {
        "w": w,
        "x": y,   # Swap x and y
        "y": x,   # Swap x and y
        "z": -z,  # Negate z (accounting for Z-axis flip)
    }


def ned_heading_to_enu_quaternion(ned_heading: float) -> typing.Dict[str, float]:
    """
    Convert NED heading to ENU yaw and then to quaternion.
    NED heading: 0 = North, positive = clockwise
    ENU yaw: 0 = East, positive = counter-clockwise
    Conversion: ENU_yaw = π/2 - NED_heading
    """
    # Convert NED heading to ENU yaw
    enu_yaw = np.pi / 2 - ned_heading
    half_yaw = enu_yaw / 2
    return {
        "w": np.cos(half_yaw),
        "x": 0,
        "y": 0,
        "z": np.sin(half_yaw),
    }


def extract_yaw(q: typing.Dict[str, float]) -> float:
    """Extract yaw (rotation around Z-axis) from quaternion"""
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw


def yaw_to_quaternion(yaw: float) -> typing.Dict[str, float]:
    """Create quaternion from yaw angle (rotation around Z-axis only)"""
    half_yaw = yaw / 2
    return {
        "w": np.cos(half_yaw),
        "x": 0,
        "y": 0,
        "z": np.sin(half_yaw),
    }


def multiply_quaternions(
    q1: typing.Dict[str, float], q2: typing.Dict[str, float],
) -> typing.Dict[str, float]:
    """Multiply two quaternions"""
    return {
        "w": q1["w"] * q2["w"] - q1["x"] * q2["x"] - q1["y"] * q2["y"] - q1["z"] * q2["z"],
        "x": q1["w"] * q2["x"] + q1["x"] * q2["w"] + q1["y"] * q2["z"] - q1["z"] * q2["y"],
        "y": q1["w"] * q2["y"] - q1["x"] * q2["z"] + q1["y"] * q2["w"] + q1["z"] * q2["x"],
        "z": q1["w"] * q2["z"] + q1["x"] * q2["y"] - q1["y"] * q2["x"] + q1["z"] * q2["w"],
    }


def quaternion_conjugate(q: typing.Dict[str, float]) -> typing.Dict[str, float]:
    """Return the conjugate (inverse rotation) of a quaternion"""
    return {
        "w": q["w"],
        "x": -q["x"],
        "y": -q["y"],
        "z": -q["z"],
    }


def extract_pitch_and_roll(q: typing.Dict[str, float]) -> typing.Dict[str, float]:
    """
    Extract pitch and roll only (remove yaw) from quaternion.
    This gives the rotation from base_intermediate (yaw-only) to base_link (full attitude)
    """
    # Extract yaw
    yaw = extract_yaw(q)

    # Create yaw-only quaternion
    yaw_q = yaw_to_quaternion(yaw)

    # Remove yaw: q_pitch_roll = q * yawQ^-1
    # This gives us the rotation that, when combined with yaw, gives the full attitude
    yaw_q_inv = quaternion_conjugate(yaw_q)
    return multiply_quaternions(q, yaw_q_inv)


def build_position_covariance(eph: typing.Optional[float],
                               epv: typing.Optional[float]) -> typing.List[float]:
    """
    Build position covariance matrix from eph and epv.
    Position covariance is in ENU (East, North, Up) frame, row-major order.
    """
    if eph is None and epv is None:
        return [0.0] * 9

    # Convert standard deviation to variance (square it)
    eph_var = (eph * eph) if eph is not None else 0.0
    epv_var = (epv * epv) if epv is not None else 0.0

    # Position covariance matrix in ENU frame (row-major):
    # [E_E  E_N  E_U]
    # [N_E  N_N  N_U]
    # [U_E  U_N  U_U]
    # For diagonal covariance: E_E = N_N = eph^2, U_U = epv^2
    return [
        eph_var, 0.0, 0.0,      # East-East, East-North, East-Up
        0.0, eph_var, 0.0,      # North-East, North-North, North-Up
        0.0, 0.0, epv_var,      # Up-East, Up-North, Up-Up
    ]


def ulog_level_to_fox_log_level(level: str) -> int:
    """Map ULog log level string to foxglove.Log level integer."""
    if level == "EMERGENCY":
        return 5  # FATAL
    if level in ("ALERT", "CRITICAL", "ERROR"):
        return 4  # ERROR
    if level in ("WARNING", "NOTICE"):
        return 3  # WARNING
    if level == "INFO":
        return 2  # INFO
    if level == "DEBUG":
        return 1  # DEBUG
    return 0  # UNKNOWN


def _message_to_dict(instance) -> dict:
    """Convert a protobuf message instance to a dict, compatible across protobuf versions."""
    signature = inspect.signature(MessageToDict)
    if 'always_print_fields_with_no_presence' in signature.parameters:
        return MessageToDict(instance, always_print_fields_with_no_presence=True)
    # pylint: disable=unexpected-keyword-arg
    return MessageToDict(instance, including_default_value_fields=True)


def get_foxglove_json_schema(schema_name: str) -> typing.Tuple[str, dict]:
    """
    Get a Foxglove JSON schema by name from the installed foxglove-sdk package.

    Args:
        schema_name: Schema name like "LocationFix", "FrameTransform", "PosesInFrame", "Log"

    Returns:
        Tuple of (full_schema_name, json_schema_dict)

    Raises:
        ValueError: If schema cannot be loaded from foxglove-sdk package
    """
    if schema_name not in FOXGLOVE_SCHEMA_CLASSES:
        raise ValueError(f"Unknown schema name: {schema_name}")

    schema_class = FOXGLOVE_SCHEMA_CLASSES[schema_name]

    # Get the schema object to get the full schema name
    schema_obj = schema_class.get_schema()
    full_schema_name = schema_obj.name

    # Create an instance and build JSON schema from it
    instance = schema_class()
    json_schema = _build_json_schema_from_instance(instance, full_schema_name)

    return full_schema_name, json_schema


def _build_json_schema_from_instance(instance, full_schema_name: str) -> dict:
    """
    Build JSON schema from a schema class instance by inspecting its fields.
    """
    # Convert the instance to a dict to see its structure
    try:
        _message_to_dict(instance)
    except (AttributeError, TypeError, ValueError):
        # Fallback: try to get fields from the descriptor
        if not hasattr(instance, 'DESCRIPTOR'):
            pass

    # Build JSON schema from the message structure
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": full_schema_name,
        "type": "object",
        "properties": {},
    }

    # Get field information from the descriptor
    if hasattr(instance, 'DESCRIPTOR'):
        descriptor = instance.DESCRIPTOR
        required_fields = []

        for field in descriptor.fields:
            field_name = field.name
            field_schema = _protobuf_field_to_json_schema(field)
            json_schema["properties"][field_name] = field_schema

            # Check if field is required (not optional and not repeated with default)
            if field.label != field.LABEL_REPEATED:
                required_fields.append(field_name)

        if required_fields:
            json_schema["required"] = required_fields

    return json_schema


def _protobuf_field_to_json_schema(field) -> dict:
    """Convert a protobuf field descriptor to a JSON schema type definition."""
    # Map protobuf types to JSON schema types
    type_map = {
        1: {"type": "number"},   # double
        2: {"type": "number"},   # float
        3: {"type": "integer"},  # int64
        4: {"type": "integer"},  # uint64
        5: {"type": "integer"},  # int32
        6: {"type": "integer"},  # fixed64
        7: {"type": "integer"},  # fixed32
        8: {"type": "boolean"},  # bool
        9: {"type": "string"},   # string
        11: {"type": "object"},  # message
        13: {"type": "integer"}, # uint32
        14: {"type": "integer"}, # enum
        15: {"type": "integer"}, # sfixed32
        16: {"type": "integer"}, # sfixed64
        17: {"type": "integer"}, # sint32
        18: {"type": "integer"}, # sint64
    }

    base_schema = type_map.get(field.type, {"type": "string"})

    # Handle repeated fields (arrays)
    if field.label == field.LABEL_REPEATED:
        return {
            "type": "array",
            "items": base_schema.copy()
        }

    # Handle message types (nested objects)
    if field.type == field.TYPE_MESSAGE:
        # For nested messages, create a basic object schema
        return {"type": "object"}

    return base_schema.copy()


def get_foxglove_location_fix_schema() -> typing.Tuple[str, dict]:
    """Return JSON schema for foxglove.LocationFix"""
    return get_foxglove_json_schema("LocationFix")


def get_foxglove_frame_transform_schema() -> typing.Tuple[str, dict]:
    """Return JSON schema for foxglove.FrameTransform"""
    return get_foxglove_json_schema("FrameTransform")


def get_foxglove_poses_in_frame_schema() -> typing.Tuple[str, dict]:
    """Return JSON schema for foxglove.PosesInFrame"""
    return get_foxglove_json_schema("PosesInFrame")


def get_foxglove_log_schema() -> typing.Tuple[str, dict]:
    """Return JSON schema for foxglove.Log"""
    return get_foxglove_json_schema("Log")


def map_to_foxglove_schema(message_name: str) -> typing.Optional[str]:
    """
    Map PX4 message name to Foxglove schema name.
    Returns None if no mapping exists.
    """
    foxglove_mappings = {
        "vehicle_global_position": "foxglove.LocationFix",
        "vehicle_local_position": "foxglove.FrameTransform",
        "vehicle_attitude": "foxglove.FrameTransform",
    }
    return foxglove_mappings.get(message_name)


def build_foxglove_schema(schema_name: str) -> typing.Tuple[str, str, bytes]:
    """
    Return JSON schema definition for a Foxglove schema.

    Returns:
        Tuple of (full_schema_name, encoding, schema_data_bytes)
    """
    schema_builders = {
        "foxglove.LocationFix": get_foxglove_location_fix_schema,
        "foxglove.FrameTransform": get_foxglove_frame_transform_schema,
        "foxglove.PosesInFrame": get_foxglove_poses_in_frame_schema,
        "foxglove.Log": get_foxglove_log_schema,
    }
    builder = schema_builders.get(schema_name)
    if builder:
        full_name, json_schema_dict = builder()
        return (full_name, "jsonschema", json.dumps(json_schema_dict).encode("utf-8"))
    raise ValueError(f"Unknown Foxglove schema: {schema_name}")


def build_json_schema(message_name: str, ulog: ULog) -> dict:
    """Build JSON schema from ULog message format (for unmapped messages)"""
    properties = {}
    for field in ulog.message_formats[message_name].fields:
        type_str, array_length, name = field
        if name == "timestamp" or name.startswith("_padding"):
            continue
        primitive_type = _get_primitive_type(type_str)
        if primitive_type is None:
            field_type = build_json_schema(type_str, ulog)
        else:
            field_type = {"type": primitive_type}
        if array_length > 0:
            if primitive_type == "string":
                field_type = {
                    "type": "string",
                    "minLength": array_length,
                    "maxLength": array_length
                }
            else:
                field_type = {
                    "type": "array",
                    "items": field_type,
                    "minItems": array_length,
                    "maxItems": array_length,
                }
        properties[name] = field_type
    return {
        "title": message_name,
        "type": "object",
        "properties": properties
    }


def _parameter_type_to_json_schema(param_value: typing.Any) -> dict:
    """Return a JSON schema type definition for a parameter value."""
    if isinstance(param_value, str):
        return {"type": "string"}
    if np.issubdtype(type(param_value), np.floating):
        return {"type": "number"}
    if np.issubdtype(type(param_value), np.integer):
        return {"type": "integer"}
    raise ValueError(f"Unsupported parameter type: {type(param_value)}")


def _parameter_value_to_python(param_value: typing.Any) -> typing.Any:
    """Convert a parameter value to a JSON-serializable Python type."""
    if np.issubdtype(type(param_value), np.floating):
        return float(param_value)
    if np.issubdtype(type(param_value), np.integer):
        return int(param_value)
    return param_value


def build_parameters_json_schema(initial_parameters: typing.Dict[str, typing.Any]) -> dict:
    """Build a JSON schema for the /parameters channel from initial parameters."""
    properties = {
        param_name: _parameter_type_to_json_schema(param_value)
        for param_name, param_value in initial_parameters.items()
    }
    return {
        "type": "object",
        "properties": properties,
    }


def _read_string_data(
    data: ULog.Data,
    field_name: str,
    array_size: int,
    data_index: int,
    disable_str_exceptions: bool = False,
) -> str:
    """Parse a data field as a null-terminated string."""
    value = ""
    for index in range(array_size):
        character = int(data.data[f"{field_name}[{index}]"][data_index])
        if character == 0:
            break
        try:
            value += chr(character)
        except ValueError:
            if not disable_str_exceptions:
                raise
    return value


def _get_value(
    data: ULog.Data,
    fieldname: str,
    schema: dict,
    idx: int,
    disable_str_exceptions: bool = False,
) -> typing.Any:
    """Extract value from ULog data based on schema"""
    if schema["type"] == "object":
        value = {}
        for prop_field, prop_field_schema in schema["properties"].items():
            nested_field = f"{fieldname}.{prop_field}"
            value[prop_field] = _get_value(
                data, nested_field, prop_field_schema, idx, disable_str_exceptions,
            )
        return value
    if schema["type"] == "array":
        array_length = schema["minItems"]
        value = []
        for i in range(array_length):
            value.append(_get_value(
                data, f"{fieldname}[{i}]", schema["items"], idx, disable_str_exceptions,
            ))
        return value
    if schema["type"] == "string":
        return _read_string_data(
            data, fieldname, schema["minLength"], idx, disable_str_exceptions,
        )

    value = data.data[fieldname][idx]
    if np.isnan(value):
        value = None
    elif schema["type"] == "integer":
        value = int(value)
    elif schema["type"] == "number":
        value = float(value)
    elif schema["type"] == "boolean":
        value = bool(value)
    return value


def convert_timestamp_to_foxglove_time(timestamp_ns: int) -> dict:
    """Convert timestamp in nanoseconds to Foxglove Time format (sec, nsec)"""
    sec = int(timestamp_ns // 1e9)
    nsec = int(timestamp_ns % 1e9)
    return {"sec": sec, "nsec": nsec}


def convert_px4_to_foxglove(message_name: str, data: ULog.Data, idx: int,
                            absolute_timestamp_ns: int) -> typing.Optional[dict]:
    """
    Convert PX4 message to Foxglove format.
    Returns None if conversion is not supported.
    """
    timestamp = convert_timestamp_to_foxglove_time(absolute_timestamp_ns)

    if message_name == "vehicle_global_position":
        lat = float(data.data.get('lat', [0])[idx])
        lon = float(data.data.get('lon', [0])[idx])
        alt = float(data.data.get('alt', [0])[idx])
        eph = data.data.get('eph', [None])[idx] if 'eph' in data.data else None
        epv = data.data.get('epv', [None])[idx] if 'epv' in data.data else None

        eph_is_nan = isinstance(eph, (int, float)) and np.isnan(eph)
        epv_is_nan = isinstance(epv, (int, float)) and np.isnan(epv)
        eph_val = float(eph) if eph is not None and not eph_is_nan else None
        epv_val = float(epv) if epv is not None and not epv_is_nan else None

        position_covariance = build_position_covariance(eph_val, epv_val)
        has_covariance = eph_val is not None or epv_val is not None
        position_covariance_type = 2 if has_covariance else 0  # 2 = DIAGONAL_KNOWN, 0 = UNKNOWN

        return {
            "timestamp": timestamp,
            "frame_id": "base_link",
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "position_covariance": position_covariance,
            "position_covariance_type": position_covariance_type,
        }

    if message_name == "vehicle_local_position":
        x = float(data.data.get('x', [0])[idx])
        y = float(data.data.get('y', [0])[idx])
        z = float(data.data.get('z', [0])[idx])
        heading = data.data.get('heading', [None])[idx] if 'heading' in data.data else None

        pos_enu = ned_to_enu_position(x, y, z)

        # Use heading to create quaternion if available, otherwise use identity rotation
        if heading is not None and not (isinstance(heading, (int, float)) and np.isnan(heading)):
            rotation = ned_heading_to_enu_quaternion(float(heading))
        else:
            rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}  # Identity quaternion

        return {
            "timestamp": timestamp,
            "parent_frame_id": "local_origin",
            "child_frame_id": "base_intermediate",  # Has no roll and pitch, only yaw
            "translation": pos_enu,
            "rotation": rotation,
        }

    if message_name == "vehicle_attitude":
        # In ULog, array fields like q[4] are stored as separate fields: q[0], q[1], q[2], q[3]
        # Check if we have the array fields
        has_q_array = all(f'q[{i}]' in data.data for i in range(4))
        if not has_q_array:
            # Fallback: try accessing as 'q' if it's stored as a structured array
            if 'q' not in data.data:
                return None
            try:
                q_data = data.data['q']
                if idx >= len(q_data):
                    return None
                q = q_data[idx]
                if isinstance(q, np.ndarray) and q.shape == (4,):
                    q_ned = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]  # [w, x, y, z]
                elif isinstance(q, (list, tuple)) and len(q) == 4:
                    q_ned = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]  # [w, x, y, z]
                else:
                    return None
            except (IndexError, TypeError, ValueError):
                return None
        else:
            # Access quaternion from separate array fields: q[0], q[1], q[2], q[3]
            # PX4 stores quaternion as [w, x, y, z]
            try:
                q_ned = [
                    float(data.data['q[0]'][idx]),  # w
                    float(data.data['q[1]'][idx]),  # x
                    float(data.data['q[2]'][idx]),  # y
                    float(data.data['q[3]'][idx]),   # z
                ]
            except (IndexError, KeyError, TypeError, ValueError):
                return None

        # Convert NED quaternion to ENU
        q_enu = ned_quaternion_to_enu(q_ned)

        # Extract pitch and roll only (remove yaw)
        # Rotation from base_intermediate (yaw-only) to base_link (full attitude)
        pitch_roll_rotation = extract_pitch_and_roll(q_enu)
        zero_translation = {"x": 0.0, "y": 0.0, "z": 0.0}

        return {
            "timestamp": timestamp,
            "parent_frame_id": "base_intermediate",
            "child_frame_id": "base_link",
            "translation": zero_translation,
            "rotation": pitch_roll_rotation,
        }

    return None


def _collect_log_message_items(
    ulog: ULog,
    log_channel_id: int,
    use_absolute_time: bool,
    reference_time_us: typing.Optional[int],
    first_relative_timestamp_us: typing.Optional[int],
) -> typing.List[typing.Tuple[int, int, dict]]:
    """Collect logged messages for the /log_message channel."""
    items = []
    if not ulog.logged_messages and not ulog.logged_messages_tagged:
        return items

    for log_msg in itertools.chain(
        ulog.logged_messages,
        *ulog.logged_messages_tagged.values(),
    ):
        timestamp_ns = relative_timestamp_to_ns(
            log_msg.timestamp,
            use_absolute_time,
            reference_time_us,
            first_relative_timestamp_us,
        )
        message = {
            "timestamp": convert_timestamp_to_foxglove_time(timestamp_ns),
            "level": ulog_level_to_fox_log_level(log_msg.log_level_str()),
            "message": log_msg.message,
        }
        items.append((log_channel_id, timestamp_ns, message))
    return items


def _collect_parameter_message_items(
    ulog: ULog,
    parameter_channel_id: int,
    use_absolute_time: bool,
    reference_time_us: typing.Optional[int],
    first_relative_timestamp_us: typing.Optional[int],
) -> typing.List[typing.Tuple[int, int, dict]]:
    """Collect parameter messages for the /parameters channel."""
    items = []
    if not ulog.initial_parameters:
        return items

    initial_timestamp_ns = relative_timestamp_to_ns(
        ulog.start_timestamp,
        use_absolute_time,
        reference_time_us,
        first_relative_timestamp_us,
    )
    initial_params = {
        param_name: _parameter_value_to_python(param_value)
        for param_name, param_value in ulog.initial_parameters.items()
    }
    items.append((parameter_channel_id, initial_timestamp_ns, initial_params))

    for timestamp, param_name, param_value in ulog.changed_parameters:
        timestamp_ns = relative_timestamp_to_ns(
            timestamp,
            use_absolute_time,
            reference_time_us,
            first_relative_timestamp_us,
        )
        items.append((
            parameter_channel_id,
            timestamp_ns,
            {param_name: _parameter_value_to_python(param_value)},
        ))
    return items


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Convert ULog to MCAP')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')
    parser.add_argument('mcap', metavar='file.mcap', help='MCAP output file')

    parser.add_argument(
        '-m', '--messages', dest='messages',
        help=("Only consider given messages. Must be a comma-separated list of"
              " names, like 'sensor_combined,vehicle_gps_position'"))

    parser.add_argument('-i', '--ignore', dest='ignore', action='store_true',
                        help='Ignore string parsing exceptions', default=False)

    parser.add_argument(
        '-d', '--metadata', metavar='key=value', action='append', type=str,
        help='Additional file-level metadata (repeatable)')

    parser.add_argument(
        '-n', '--metadata-name', dest='metadata_name', type=str,
        default='ulog-metadata',
        help='Name for metadata group, if adding metadata')

    args = parser.parse_args()

    metadata = None
    if args.metadata:
        metadata_dict = {}
        for item in args.metadata:
            key, value = item.split('=', 1)
            metadata_dict[key] = value
        metadata = [(args.metadata_name, metadata_dict)]

    convert_ulog2mcap(args.filename, args.mcap, args.messages, args.ignore, metadata)


def convert_ulog2mcap(
    ulog_file_name: str,
    mcap_file_name: str,
    messages: typing.Optional[str],
    disable_str_exceptions: bool = False,
    metadata: typing.Optional[typing.List[typing.Tuple[str, typing.Dict[str, str]]]] = None,
):
    """
    Convert a ULog file to MCAP format using Foxglove schemas.

    A single flight path (PosesInFrame) containing all segments the aircraft flew
    is always generated and published on /px4/flight_path at the beginning of
    the recording.

    :param ulog_file_name: The ULog filename to open and read
    :param mcap_file_name: The MCAP filename to open and write
    :param messages: A comma-separated list of message names to filter
    :param disable_str_exceptions: Ignore string parsing exceptions
    :param metadata: Optional file-level metadata as (name, dict) tuples
    """
    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list

    # Get absolute timestamp reference
    reference_time_us, first_relative_timestamp_us = get_absolute_timestamp_reference(ulog)
    use_absolute_time = reference_time_us is not None

    if not use_absolute_time:
        print("Warning: No absolute time reference found. Using relative timestamps.")

    multiids = defaultdict(set)
    for d in data:
        multiids[d.name].add(d.multi_id)

    schemas = {}  # Map message_name -> schema_id
    channels = {}  # Map (message_name, multi_id) -> channel_id

    # Path tracking: accumulate full flight path (all segments), published once at start
    last_path_position = None  # Last position where we dropped a point (ENU coordinates)
    path_poses = []  # All poses for the final path message

    with open(mcap_file_name, "wb") as stream:
        mcap = Writer(stream, compression=CompressionType.LZ4)
        mcap.start()

        for name, metadata_dict in metadata or []:
            mcap.add_metadata(name, metadata_dict)

        # Always register path channel (PosesInFrame) for full flight path
        path_schema_name = "foxglove.PosesInFrame"
        path_full_schema_name, path_encoding, path_schema_data = build_foxglove_schema(
            path_schema_name,
        )
        path_schema_id = mcap.register_schema(
            name=path_full_schema_name,
            encoding=path_encoding,
            data=path_schema_data,
        )
        path_channel_id = mcap.register_channel(
            schema_id=path_schema_id,
            topic="/px4/flight_path",
            message_encoding="json",
        )

        log_channel_id = None
        if ulog.logged_messages or ulog.logged_messages_tagged:
            log_full_schema_name, log_encoding, log_schema_data = build_foxglove_schema(
                "foxglove.Log",
            )
            log_schema_id = mcap.register_schema(
                name=log_full_schema_name,
                encoding=log_encoding,
                data=log_schema_data,
            )
            log_channel_id = mcap.register_channel(
                schema_id=log_schema_id,
                topic="/log_message",
                message_encoding="json",
            )

        parameter_channel_id = None
        if ulog.initial_parameters:
            parameter_schema = build_parameters_json_schema(ulog.initial_parameters)
            parameter_schema_id = mcap.register_schema(
                name="Parameters",
                encoding="jsonschema",
                data=json.dumps(parameter_schema).encode("utf-8"),
            )
            parameter_channel_id = mcap.register_channel(
                schema_id=parameter_schema_id,
                topic="/parameters",
                message_encoding="json",
            )

        # First pass: register all schemas and channels
        for d in data:
            if d.name in schemas:
                schema_id = schemas[d.name]
            else:
                # Check if we should use Foxglove schema
                foxglove_schema_name = map_to_foxglove_schema(d.name)
                if foxglove_schema_name:
                    full_schema_name, encoding, schema_data = build_foxglove_schema(
                        foxglove_schema_name,
                    )
                    schema_id = mcap.register_schema(
                        name=full_schema_name,
                        encoding=encoding,
                        data=schema_data,
                    )
                    schemas[d.name] = schema_id
                else:
                    # Build JSON schema from ULog message format
                    schema = build_json_schema(d.name, ulog)
                    schema_id = mcap.register_schema(
                        name=d.name,
                        encoding="jsonschema",
                        data=json.dumps(schema).encode("utf-8"),
                    )
                    schemas[d.name] = schema_id

            # Register channel
            channel_key = (d.name, d.multi_id)
            if channel_key not in channels:
                if multiids[d.name] == {0}:
                    topic = "/px4/{}".format(d.name)
                else:
                    topic = "/px4/{}_{}".format(d.name, d.multi_id)

                channel_id = mcap.register_channel(
                    schema_id=schema_id,
                    topic=topic,
                    message_encoding="json",
                )
                channels[channel_key] = channel_id

        # Second pass: write messages and accumulate path poses
        items = []
        for d in data:
            channel_id = channels[(d.name, d.multi_id)]
            foxglove_schema_name = map_to_foxglove_schema(d.name)

            num_messages = len(d.data['timestamp'])
            for idx in range(num_messages):
                relative_timestamp_us = int(d.data['timestamp'][idx])

                absolute_timestamp_ns = relative_timestamp_to_ns(
                    relative_timestamp_us,
                    use_absolute_time,
                    reference_time_us,
                    first_relative_timestamp_us,
                )

                # Convert to Foxglove format if mapped
                if foxglove_schema_name:
                    message = convert_px4_to_foxglove(d.name, d, idx, absolute_timestamp_ns)
                    if message is None:
                        # Fallback to generic JSON schema
                        schema = build_json_schema(d.name, ulog)
                        message = {}
                        for field, field_schema in schema["properties"].items():
                            message[field] = _get_value(
                                d, field, field_schema, idx, disable_str_exceptions,
                            )
                else:
                    # Use generic JSON schema
                    schema = build_json_schema(d.name, ulog)
                    message = {}
                    for field, field_schema in schema["properties"].items():
                        message[field] = _get_value(
                            d, field, field_schema, idx, disable_str_exceptions,
                        )

                items.append((channel_id, absolute_timestamp_ns, message))

                # Accumulate flight path from vehicle_local_position
                if d.name == "vehicle_local_position":
                    x = float(d.data.get('x', [0])[idx])
                    y = float(d.data.get('y', [0])[idx])
                    z = float(d.data.get('z', [0])[idx])
                    heading = d.data.get('heading', [None])[idx] if 'heading' in d.data else None

                    # Convert to ENU coordinates (local coordinate system)
                    pos_enu = ned_to_enu_position(x, y, z)

                    # Drop a point when moved PATH_SEGMENT_LENGTH or more
                    should_drop_point = False
                    if last_path_position is None:
                        should_drop_point = True
                    else:
                        dx = pos_enu["x"] - last_path_position["x"]
                        dy = pos_enu["y"] - last_path_position["y"]
                        dz = pos_enu["z"] - last_path_position["z"]
                        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                        if distance >= PATH_SEGMENT_LENGTH:
                            should_drop_point = True

                    if should_drop_point:
                        heading_is_nan = isinstance(heading, (int, float)) and np.isnan(heading)
                        if heading is not None and not heading_is_nan:
                            rotation = ned_heading_to_enu_quaternion(float(heading))
                        else:
                            rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                        path_poses.append({
                            "position": pos_enu,
                            "orientation": rotation,
                        })
                        last_path_position = pos_enu.copy()

        if log_channel_id is not None:
            items.extend(_collect_log_message_items(
                ulog,
                log_channel_id,
                use_absolute_time,
                reference_time_us,
                first_relative_timestamp_us,
            ))

        if parameter_channel_id is not None:
            items.extend(_collect_parameter_message_items(
                ulog,
                parameter_channel_id,
                use_absolute_time,
                reference_time_us,
                first_relative_timestamp_us,
            ))

        # Publish full flight path once at the beginning of the recording.
        # Use the minimum message timestamp (valid MCAP uint64, non-negative).
        start_timestamp_ns = min((t for (_, t, _) in items), default=0)
        start_timestamp_ns = max(0, min(MCAP_TIMESTAMP_MAX, int(start_timestamp_ns)))
        path_message = {
            "timestamp": convert_timestamp_to_foxglove_time(start_timestamp_ns),
            "frame_id": "local_origin",
            "poses": path_poses,
        }
        items.append((path_channel_id, start_timestamp_ns, path_message))

        # Sort by timestamp and write (timestamps already clamped to MCAP uint64 range)
        items.sort(key=lambda x: x[1])
        for channel_id, timestamp_ns, message in items:
            ts = max(0, min(MCAP_TIMESTAMP_MAX, int(timestamp_ns)))
            mcap.add_message(
                channel_id,
                log_time=ts,
                publish_time=ts,
                data=json.dumps(message).encode("utf-8")
            )

        mcap.finish()

    print(f"Converted {ulog_file_name} to {mcap_file_name}")
