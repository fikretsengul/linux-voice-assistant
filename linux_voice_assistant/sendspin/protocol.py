"""Sendspin Protocol message types and serialization.

This module implements the Sendspin Protocol message format for WebSocket
communication with Music Assistant and other Sendspin servers.

Protocol Reference: https://github.com/Sendspin/spec
"""

import json
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PROTOCOL_VERSION = 1
RECOMMENDED_PORT = 8927
RECOMMENDED_PATH = "/sendspin"

# Binary message type IDs (from spec)
# Player role uses IDs 4-7 (bits 000001xx)
BINARY_TYPE_AUDIO_CHUNK = 4

# Artwork role uses IDs 8-11 (bits 000010xx)
BINARY_TYPE_ARTWORK_CHANNEL_0 = 8
BINARY_TYPE_ARTWORK_CHANNEL_1 = 9
BINARY_TYPE_ARTWORK_CHANNEL_2 = 10
BINARY_TYPE_ARTWORK_CHANNEL_3 = 11

# Visualizer role uses IDs 16-23 (bits 00010xxx)
BINARY_TYPE_VISUALIZER = 16


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class ClientSyncState(str):
    """Client operational state for player role."""
    SYNCHRONIZED = "synchronized"
    ERROR = "error"
    EXTERNAL_SOURCE = "external_source"


class PlaybackState(str):
    """Group playback state."""
    PLAYING = "playing"
    STOPPED = "stopped"


class PlayerCommand(str):
    """Commands that can be sent to a player."""
    VOLUME = "volume"
    MUTE = "mute"


class ControllerCommand(str):
    """Commands that a controller can send."""
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    VOLUME = "volume"
    MUTE = "mute"
    SWITCH = "switch"


# -----------------------------------------------------------------------------
# Audio Format Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class AudioFormat:
    """Describes an audio format supported by the player."""
    codec: str  # "opus", "flac", "pcm"
    channels: int  # e.g., 2 for stereo
    sample_rate: int  # e.g., 48000
    bit_depth: int  # e.g., 16, 24

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "codec": self.codec,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
        }


# -----------------------------------------------------------------------------
# Client Messages (Client -> Server)
# -----------------------------------------------------------------------------


@dataclass
class ClientHello:
    """client/hello message sent after WebSocket connection established.

    This message identifies the client and declares its capabilities.
    """
    client_id: str
    name: str
    supported_roles: List[str] = field(default_factory=lambda: ["player@v1"])
    version: int = PROTOCOL_VERSION
    device_info: Optional[Dict[str, str]] = None

    # Player role support (if player@v1 in supported_roles)
    supported_formats: List[AudioFormat] = field(default_factory=list)
    buffer_capacity: int = 2_000_000  # bytes
    supported_commands: List[str] = field(
        default_factory=lambda: ["volume", "mute"]
    )

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket text message."""
        payload: Dict[str, Any] = {
            "client_id": self.client_id,
            "name": self.name,
            "version": self.version,
            "supported_roles": self.supported_roles,
        }

        if self.device_info:
            payload["device_info"] = self.device_info

        # Add player support if player role is declared
        # Note: Music Assistant's aiosendspin expects "player_support" not "player@v1_support"
        if "player@v1" in self.supported_roles:
            payload["player_support"] = {
                "supported_formats": [f.to_dict() for f in self.supported_formats],
                "buffer_capacity": self.buffer_capacity,
                "supported_commands": self.supported_commands,
            }

        return json.dumps({"type": "client/hello", "payload": payload})


@dataclass
class ClientTime:
    """client/time message for clock synchronization.

    Sent periodically to establish clock offset between client and server.
    """
    client_transmitted: int  # Client's internal clock in microseconds

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket text message."""
        return json.dumps({
            "type": "client/time",
            "payload": {
                "client_transmitted": self.client_transmitted,
            }
        })


@dataclass
class ClientState:
    """client/state message reporting player state to server.

    Sent after server/hello and whenever state changes.
    """
    state: str = ClientSyncState.SYNCHRONIZED
    volume: Optional[int] = None  # 0-100 perceived loudness
    muted: Optional[bool] = None

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket text message."""
        payload: Dict[str, Any] = {"state": self.state}

        # Player-specific state
        player_state: Dict[str, Any] = {}
        if self.volume is not None:
            player_state["volume"] = self.volume
        if self.muted is not None:
            player_state["muted"] = self.muted

        if player_state:
            payload["player"] = player_state

        return json.dumps({"type": "client/state", "payload": payload})


@dataclass
class ClientGoodbye:
    """client/goodbye message sent before graceful disconnect."""
    reason: str  # "another_server", "shutdown", "restart", "user_request"

    def to_json(self) -> str:
        """Serialize to JSON for WebSocket text message."""
        return json.dumps({
            "type": "client/goodbye",
            "payload": {"reason": self.reason}
        })


# -----------------------------------------------------------------------------
# Server Messages (Server -> Client)
# -----------------------------------------------------------------------------


@dataclass
class ServerHello:
    """server/hello message received after client/hello."""
    server_id: str
    name: str
    version: int
    active_roles: List[str]
    connection_reason: str  # "discovery" or "playback"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerHello":
        """Parse from JSON payload dictionary."""
        return cls(
            server_id=data.get("server_id", ""),
            name=data.get("name", ""),
            version=data.get("version", 1),
            active_roles=data.get("active_roles", []),
            connection_reason=data.get("connection_reason", "discovery"),
        )


@dataclass
class ServerTime:
    """server/time message for clock synchronization."""
    client_transmitted: int  # Echo of client's timestamp
    server_received: int  # When server received client/time
    server_transmitted: int  # When server sent this response

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerTime":
        """Parse from JSON payload dictionary."""
        return cls(
            client_transmitted=data.get("client_transmitted", 0),
            server_received=data.get("server_received", 0),
            server_transmitted=data.get("server_transmitted", 0),
        )


@dataclass
class StreamStart:
    """stream/start message indicating audio stream configuration."""
    codec: str
    sample_rate: int
    channels: int
    bit_depth: int
    codec_header: Optional[bytes] = None  # Base64 decoded header for FLAC

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamStart":
        """Parse from player object in stream/start payload."""
        import base64

        codec_header = None
        if "codec_header" in data:
            codec_header = base64.b64decode(data["codec_header"])

        return cls(
            codec=data.get("codec", "opus"),
            sample_rate=data.get("sample_rate", 48000),
            channels=data.get("channels", 2),
            bit_depth=data.get("bit_depth", 16),
            codec_header=codec_header,
        )


@dataclass
class ServerCommand:
    """server/command message requesting player action."""
    command: str  # "volume" or "mute"
    volume: Optional[int] = None  # For "volume" command, 0-100
    mute: Optional[bool] = None  # For "mute" command

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerCommand":
        """Parse from player object in server/command payload."""
        return cls(
            command=data.get("command", ""),
            volume=data.get("volume"),
            mute=data.get("mute"),
        )


@dataclass
class GroupUpdate:
    """group/update message with group state changes."""
    playback_state: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupUpdate":
        """Parse from JSON payload dictionary."""
        return cls(
            playback_state=data.get("playback_state"),
            group_id=data.get("group_id"),
            group_name=data.get("group_name"),
        )


# -----------------------------------------------------------------------------
# Binary Message Parsing
# -----------------------------------------------------------------------------


@dataclass
class AudioChunk:
    """Parsed binary audio chunk message."""
    timestamp_us: int  # Server clock time when first sample should play
    data: bytes  # Encoded audio frame


def parse_binary_message(data: bytes) -> Tuple[int, int, bytes]:
    """Parse a binary WebSocket message.

    Binary message format:
    - Byte 0: message type (uint8)
    - Bytes 1-8: timestamp (big-endian int64, microseconds)
    - Remaining bytes: payload

    Args:
        data: Raw binary message bytes

    Returns:
        Tuple of (message_type, timestamp_us, payload)
    """
    if len(data) < 9:
        raise ValueError(f"Binary message too short: {len(data)} bytes")

    msg_type = data[0]
    timestamp_us = struct.unpack(">q", data[1:9])[0]
    payload = data[9:]

    return msg_type, timestamp_us, payload


def parse_audio_chunk(data: bytes) -> AudioChunk:
    """Parse a binary audio chunk message.

    Args:
        data: Raw binary message bytes (including type byte)

    Returns:
        AudioChunk with timestamp and encoded audio data
    """
    msg_type, timestamp_us, payload = parse_binary_message(data)

    if msg_type != BINARY_TYPE_AUDIO_CHUNK:
        raise ValueError(f"Expected audio chunk (type 4), got type {msg_type}")

    return AudioChunk(timestamp_us=timestamp_us, data=payload)


# -----------------------------------------------------------------------------
# JSON Message Parsing
# -----------------------------------------------------------------------------


def parse_json_message(text: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a JSON WebSocket text message.

    Args:
        text: JSON string

    Returns:
        Tuple of (message_type, payload_dict)
    """
    msg = json.loads(text)
    msg_type = msg.get("type", "")
    payload = msg.get("payload", {})
    return msg_type, payload


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def get_local_time_us() -> int:
    """Get current local time in microseconds using monotonic clock."""
    return int(time.monotonic() * 1_000_000)


def create_client_id(name: str) -> str:
    """Create a persistent client ID from device name.

    The client_id should remain stable across reconnections so servers
    can associate clients with previous sessions.
    """
    import hashlib
    import uuid

    # Use machine ID if available, otherwise generate based on name
    try:
        with open("/etc/machine-id", "r") as f:
            machine_id = f.read().strip()
    except (FileNotFoundError, PermissionError):
        # Fallback: hash the name with a UUID namespace
        machine_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

    # Create stable hash from machine ID and name
    h = hashlib.sha256(f"{machine_id}:{name}".encode()).hexdigest()
    return h[:32]
