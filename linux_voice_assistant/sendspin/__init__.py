"""Sendspin Protocol client for synchronized multi-room audio.

This package implements a Sendspin Protocol client for receiving and
playing synchronized audio from Music Assistant or other Sendspin servers.

Example usage:
    from linux_voice_assistant.sendspin import SendspinClient

    client = SendspinClient(
        server_url="ws://192.168.1.100:8927/sendspin",
        client_name="Living Room Speaker",
    )
    await client.connect()

For more information about the Sendspin Protocol, see:
https://github.com/Sendspin/spec
"""

from .audio_player import SendspinAudioPlayer, SyncState
from .client import ConnectionState, SendspinClient
from .clock_sync import ClockSyncStats, KalmanClockSync
from .decoder import AudioDecoder, get_decoder, is_codec_available
from .protocol import (
    AudioFormat,
    ClientHello,
    ClientState,
    ClientTime,
    RECOMMENDED_PATH,
    RECOMMENDED_PORT,
)

__all__ = [
    # Main client
    "SendspinClient",
    "ConnectionState",
    # Audio player
    "SendspinAudioPlayer",
    "SyncState",
    # Clock synchronization
    "KalmanClockSync",
    "ClockSyncStats",
    # Decoders
    "AudioDecoder",
    "get_decoder",
    "is_codec_available",
    # Protocol types
    "AudioFormat",
    "ClientHello",
    "ClientState",
    "ClientTime",
    # Constants
    "RECOMMENDED_PORT",
    "RECOMMENDED_PATH",
]
