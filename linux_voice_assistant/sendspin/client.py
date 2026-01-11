"""Sendspin Protocol WebSocket client.

This module implements a WebSocket client that connects to a Sendspin server
(e.g., Music Assistant) and handles the protocol for synchronized audio playback.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

from .audio_player import SendspinAudioPlayer, SyncState
from .clock_sync import KalmanClockSync
from .protocol import (
    AudioFormat,
    ClientGoodbye,
    ClientHello,
    ClientState,
    ClientSyncState,
    ClientTime,
    GroupUpdate,
    BINARY_TYPE_AUDIO_CHUNK,
    ServerCommand,
    ServerHello,
    ServerTime,
    StreamStart,
    create_client_id,
    get_local_time_us,
    parse_audio_chunk,
    parse_json_message,
)

_LOGGER = logging.getLogger(__name__)

# Try to import websockets
try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    _LOGGER.warning("websockets not installed - Sendspin client unavailable")


class ConnectionState:
    """Connection state constants."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class SendspinClient:
    """Sendspin Protocol WebSocket client.

    Connects to a Sendspin server and handles:
    - Clock synchronization
    - Stream configuration
    - Audio chunk reception and playback
    - Volume/mute commands

    Designed for Music Assistant integration.
    """

    def __init__(
        self,
        server_url: str,
        client_name: str,
        output_device: Optional[str] = None,
        preferred_codec: str = "opus",
        buffer_capacity_ms: int = 2000,
        reconnect_delay: float = 5.0,
        clock_sync_interval: float = 5.0,
        on_connection_change: Optional[Callable[[str], None]] = None,
        on_volume_change: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize the Sendspin client.

        Args:
            server_url: WebSocket URL (e.g., "ws://192.168.1.100:8927/sendspin")
            client_name: Friendly name shown in Music Assistant
            output_device: Audio output device (None for default)
            preferred_codec: Preferred audio codec ("opus", "flac", "pcm")
            buffer_capacity_ms: Audio buffer size in milliseconds
            reconnect_delay: Seconds between reconnection attempts
            clock_sync_interval: Seconds between time sync messages
            on_connection_change: Callback when connection state changes
            on_volume_change: Callback when volume is changed by server
        """
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "SendspinClient requires websockets. "
                "Install with: pip install websockets"
            )

        self._server_url = server_url
        self._client_name = client_name
        self._output_device = output_device
        self._preferred_codec = preferred_codec
        self._buffer_capacity_ms = buffer_capacity_ms
        self._reconnect_delay = reconnect_delay
        self._clock_sync_interval = clock_sync_interval
        self._on_connection_change = on_connection_change
        self._on_volume_change = on_volume_change

        # Generate persistent client ID
        self._client_id = create_client_id(client_name)

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._websocket: Optional["WebSocketClientProtocol"] = None
        self._should_reconnect = True

        # Clock synchronization
        self._clock_sync = KalmanClockSync()
        self._clock_sync_task: Optional[asyncio.Task] = None

        # Audio player
        self._audio_player: Optional[SendspinAudioPlayer] = None

        # Server info
        self._server_id: Optional[str] = None
        self._server_name: Optional[str] = None
        self._active_roles: list = []

        # Group info
        self._group_id: Optional[str] = None
        self._group_name: Optional[str] = None
        self._playback_state: Optional[str] = None

        # Tasks
        self._message_task: Optional[asyncio.Task] = None
        self._connect_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> str:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected to the server."""
        return self._state in (ConnectionState.CONNECTED, ConnectionState.STREAMING)

    @property
    def volume(self) -> int:
        """Current volume level (0-100)."""
        if self._audio_player:
            return self._audio_player.volume
        return 100

    @property
    def muted(self) -> bool:
        """Current mute state."""
        if self._audio_player:
            return self._audio_player.muted
        return False

    @property
    def sync_state(self) -> str:
        """Audio synchronization state."""
        if self._audio_player:
            return self._audio_player.sync_state
        return SyncState.SYNCING

    @property
    def server_name(self) -> Optional[str]:
        """Name of the connected server."""
        return self._server_name

    @property
    def group_name(self) -> Optional[str]:
        """Name of the current group."""
        return self._group_name

    async def connect(self) -> None:
        """Connect to the Sendspin server.

        This method runs the connection loop with automatic reconnection.
        """
        self._should_reconnect = True
        await self._connection_loop()

    async def disconnect(self) -> None:
        """Disconnect from the server gracefully."""
        self._should_reconnect = False

        # Send goodbye message
        if self._websocket and not self._websocket.closed:
            try:
                goodbye = ClientGoodbye(reason="shutdown")
                await self._websocket.send(goodbye.to_json())
            except Exception:
                pass

        # Stop tasks
        if self._clock_sync_task:
            self._clock_sync_task.cancel()
            try:
                await self._clock_sync_task
            except asyncio.CancelledError:
                pass

        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass

        # Stop audio player
        if self._audio_player:
            self._audio_player.stop_playback()

        # Close WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        self._set_state(ConnectionState.DISCONNECTED)

    def set_volume(self, volume: int) -> None:
        """Set volume level locally.

        Args:
            volume: Volume 0-100
        """
        if self._audio_player:
            self._audio_player.set_volume(volume)

    def set_muted(self, muted: bool) -> None:
        """Set mute state locally.

        Args:
            muted: True to mute
        """
        if self._audio_player:
            self._audio_player.set_muted(muted)

    async def _connection_loop(self) -> None:
        """Main connection loop with reconnection logic."""
        while self._should_reconnect:
            try:
                await self._connect_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error("Connection error: %s", e)
                self._set_state(ConnectionState.ERROR)

            if self._should_reconnect:
                _LOGGER.info(
                    "Reconnecting in %.1f seconds...",
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_once(self) -> None:
        """Attempt a single connection to the server."""
        self._set_state(ConnectionState.CONNECTING)
        _LOGGER.info("Connecting to Sendspin server: %s", self._server_url)

        try:
            async with websockets.connect(
                self._server_url,
                ping_interval=30,
                ping_timeout=10,
            ) as websocket:
                self._websocket = websocket

                # Perform handshake
                await self._handshake()

                # Start clock sync task
                self._clock_sync_task = asyncio.create_task(
                    self._clock_sync_loop()
                )

                # Run message loop
                await self._message_loop()

        except websockets.exceptions.ConnectionClosed as e:
            _LOGGER.warning("Connection closed: %s", e)
        except Exception as e:
            _LOGGER.error("WebSocket error: %s", e)
            raise
        finally:
            self._websocket = None
            if self._audio_player:
                self._audio_player.stop_playback()

    async def _handshake(self) -> None:
        """Perform Sendspin handshake (client/hello -> server/hello)."""
        # Build supported formats (prefer user's choice, include fallbacks)
        formats = []
        seen_codecs = set()

        # Add preferred codec first
        if self._preferred_codec == "opus":
            formats.append(AudioFormat(
                codec="opus",
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ))
            seen_codecs.add("opus")
        elif self._preferred_codec == "flac":
            formats.append(AudioFormat(
                codec="flac",
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ))
            seen_codecs.add("flac")

        # Add fallbacks (only if not already added)
        if "opus" not in seen_codecs:
            formats.append(AudioFormat(
                codec="opus",
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ))
        if "pcm" not in seen_codecs:
            formats.append(AudioFormat(
                codec="pcm",
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ))

        # Send client/hello
        hello = ClientHello(
            client_id=self._client_id,
            name=self._client_name,
            supported_roles=["player@v1"],
            device_info={
                "product_name": "Linux Voice Assistant",
                "manufacturer": "Open Home Foundation",
                "software_version": "1.0.0",
            },
            supported_formats=formats,
            # buffer_capacity in bytes: 48000 samples/sec * 2 channels * 2 bytes * (ms/1000)
            buffer_capacity=int(48000 * 2 * 2 * self._buffer_capacity_ms / 1000),
            supported_commands=["volume", "mute"],
        )

        hello_json = hello.to_json()
        _LOGGER.debug("Sending client/hello: %s", hello_json)
        await self._websocket.send(hello_json)
        _LOGGER.debug("Sent client/hello, waiting for server/hello...")

        # Wait for server/hello with timeout
        try:
            response = await asyncio.wait_for(
                self._websocket.recv(),
                timeout=10.0
            )
            _LOGGER.debug("Received response: %s", response[:500] if isinstance(response, str) else response[:100])
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for server/hello (10s)")

        msg_type, payload = parse_json_message(response)

        if msg_type != "server/hello":
            raise RuntimeError(f"Expected server/hello, got {msg_type}")

        server_hello = ServerHello.from_dict(payload)
        self._server_id = server_hello.server_id
        self._server_name = server_hello.name
        self._active_roles = server_hello.active_roles

        _LOGGER.info(
            "Connected to %s (roles: %s)",
            self._server_name,
            ", ".join(self._active_roles),
        )

        # Initialize audio player if player role is active
        if "player@v1" in self._active_roles:
            self._audio_player = SendspinAudioPlayer(
                clock_sync=self._clock_sync,
                output_device=self._output_device,
                buffer_capacity_us=self._buffer_capacity_ms * 1000,
                on_state_change=self._on_sync_state_change,
            )

        # Send initial client/state
        await self._send_state()

        self._set_state(ConnectionState.CONNECTED)

    async def _message_loop(self) -> None:
        """Main message processing loop."""
        async for message in self._websocket:
            try:
                if isinstance(message, bytes):
                    await self._handle_binary_message(message)
                else:
                    await self._handle_text_message(message)
            except Exception as e:
                _LOGGER.warning("Error handling message: %s", e)

    async def _handle_text_message(self, text: str) -> None:
        """Handle a JSON text message from the server."""
        msg_type, payload = parse_json_message(text)

        if msg_type == "server/time":
            self._handle_time_response(payload)

        elif msg_type == "stream/start":
            await self._handle_stream_start(payload)

        elif msg_type == "stream/end":
            await self._handle_stream_end(payload)

        elif msg_type == "stream/clear":
            self._handle_stream_clear(payload)

        elif msg_type == "server/command":
            await self._handle_server_command(payload)

        elif msg_type == "group/update":
            self._handle_group_update(payload)

        elif msg_type == "server/state":
            self._handle_server_state(payload)

        else:
            _LOGGER.debug("Unhandled message type: %s", msg_type)

    async def _handle_binary_message(self, data: bytes) -> None:
        """Handle a binary message from the server."""
        if len(data) < 9:
            return

        msg_type = data[0]

        if msg_type == BINARY_TYPE_AUDIO_CHUNK:
            chunk = parse_audio_chunk(data)
            if self._audio_player:
                self._audio_player.receive_chunk(chunk.timestamp_us, chunk.data)

    def _handle_time_response(self, payload: Dict[str, Any]) -> None:
        """Handle server/time response for clock synchronization."""
        server_time = ServerTime.from_dict(payload)
        client_received_us = get_local_time_us()

        self._clock_sync.update(
            client_transmitted_us=server_time.client_transmitted,
            server_received_us=server_time.server_received,
            server_transmitted_us=server_time.server_transmitted,
            client_received_us=client_received_us,
        )

        _LOGGER.debug(
            "Clock sync: offset=%.0f us, quality=%s",
            self._clock_sync.offset_us,
            self._clock_sync.sync_quality,
        )

    async def _handle_stream_start(self, payload: Dict[str, Any]) -> None:
        """Handle stream/start message."""
        if "player" not in payload:
            return

        stream_config = StreamStart.from_dict(payload["player"])

        if self._audio_player:
            self._audio_player.configure_stream(
                codec=stream_config.codec,
                sample_rate=stream_config.sample_rate,
                channels=stream_config.channels,
                bit_depth=stream_config.bit_depth,
                codec_header=stream_config.codec_header,
            )
            self._audio_player.start_playback()
            self._set_state(ConnectionState.STREAMING)

        _LOGGER.info(
            "Stream started: %s, %d Hz",
            stream_config.codec,
            stream_config.sample_rate,
        )

    async def _handle_stream_end(self, payload: Dict[str, Any]) -> None:
        """Handle stream/end message."""
        roles = payload.get("roles", ["player"])

        if "player" in roles and self._audio_player:
            self._audio_player.stop_playback()
            self._set_state(ConnectionState.CONNECTED)

        _LOGGER.info("Stream ended")

    def _handle_stream_clear(self, payload: Dict[str, Any]) -> None:
        """Handle stream/clear message (e.g., after seek)."""
        roles = payload.get("roles", ["player"])

        if "player" in roles and self._audio_player:
            self._audio_player.clear_buffer()

        _LOGGER.debug("Stream cleared")

    async def _handle_server_command(self, payload: Dict[str, Any]) -> None:
        """Handle server/command message."""
        if "player" not in payload:
            return

        cmd = ServerCommand.from_dict(payload["player"])

        if cmd.command == "volume" and cmd.volume is not None:
            if self._audio_player:
                self._audio_player.set_volume(cmd.volume)
            if self._on_volume_change:
                self._on_volume_change(cmd.volume)
            _LOGGER.info("Volume set to %d", cmd.volume)

        elif cmd.command == "mute" and cmd.mute is not None:
            if self._audio_player:
                self._audio_player.set_muted(cmd.mute)
            _LOGGER.info("Muted: %s", cmd.mute)

        # Send updated state
        await self._send_state()

    def _handle_group_update(self, payload: Dict[str, Any]) -> None:
        """Handle group/update message."""
        update = GroupUpdate.from_dict(payload)

        if update.group_id:
            self._group_id = update.group_id
        if update.group_name:
            self._group_name = update.group_name
        if update.playback_state:
            self._playback_state = update.playback_state

        _LOGGER.debug(
            "Group update: %s, state=%s",
            self._group_name,
            self._playback_state,
        )

    def _handle_server_state(self, payload: Dict[str, Any]) -> None:
        """Handle server/state message."""
        # Currently we only implement player role, so nothing to do here
        pass

    async def _send_state(self) -> None:
        """Send current client state to server."""
        if not self._websocket or self._websocket.closed:
            return

        # Determine sync state
        if self._audio_player:
            if self._audio_player.sync_state == SyncState.SYNCHRONIZED:
                sync = ClientSyncState.SYNCHRONIZED
            elif self._audio_player.sync_state == SyncState.ERROR:
                sync = ClientSyncState.ERROR
            else:
                sync = ClientSyncState.SYNCHRONIZED
            volume = self._audio_player.volume
            muted = self._audio_player.muted
        else:
            sync = ClientSyncState.SYNCHRONIZED
            volume = 100
            muted = False

        state = ClientState(
            state=sync,
            volume=volume,
            muted=muted,
        )

        await self._websocket.send(state.to_json())

    async def _clock_sync_loop(self) -> None:
        """Periodically send time sync requests."""
        # Send a burst of time messages initially for quick sync
        for _ in range(5):
            await self._send_time()
            await asyncio.sleep(0.1)

        # Then periodic updates
        while True:
            await asyncio.sleep(self._clock_sync_interval)
            await self._send_time()

    async def _send_time(self) -> None:
        """Send a client/time message."""
        if not self._websocket or self._websocket.closed:
            return

        time_msg = ClientTime(client_transmitted=get_local_time_us())
        await self._websocket.send(time_msg.to_json())

    def _set_state(self, state: str) -> None:
        """Update connection state and notify callback."""
        if state != self._state:
            self._state = state
            _LOGGER.info("Sendspin state: %s", state)
            if self._on_connection_change:
                try:
                    self._on_connection_change(state)
                except Exception as e:
                    _LOGGER.warning("Connection callback error: %s", e)

    def _on_sync_state_change(self, state: str) -> None:
        """Handle audio sync state changes."""
        _LOGGER.debug("Sync state: %s", state)
        # Could send updated client/state here if needed
