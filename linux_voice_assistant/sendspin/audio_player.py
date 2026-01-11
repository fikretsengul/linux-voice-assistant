"""Synchronized audio player for Sendspin Protocol.

This module implements buffered, synchronized audio playback using the
timestamps provided by the Sendspin server. Audio chunks are buffered
and played at the correct time to maintain synchronization with other
players in the multi-room audio group.
"""

import logging
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Union

from .clock_sync import KalmanClockSync
from .decoder import AudioDecoder, get_decoder

_LOGGER = logging.getLogger(__name__)

# Check for mpv availability (preferred for PulseAudio/container setups)
MPV_AVAILABLE = shutil.which("mpv") is not None

# Try to import sounddevice (fallback)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class MpvOutputStream:
    """Audio output stream using mpv subprocess.

    This provides better compatibility with PulseAudio in containers
    where sounddevice/PortAudio may not work properly.
    """

    def __init__(
        self,
        device: Optional[str],
        samplerate: int,
        channels: int,
    ) -> None:
        """Initialize mpv output stream.

        Args:
            device: Audio device in mpv format (e.g., "pulse/bluez_output.xxx")
            samplerate: Sample rate in Hz
            channels: Number of audio channels
        """
        self._device = device
        self._samplerate = samplerate
        self._channels = channels
        self._proc: Optional[subprocess.Popen] = None
        self._bytes_written: int = 0

    def __enter__(self) -> "MpvOutputStream":
        """Start mpv process."""
        cmd = [
            "mpv",
            "--no-terminal",
            "--no-video",
            "--no-cache",
            "--audio-buffer=0",
            "--demuxer=rawaudio",
            f"--demuxer-rawaudio-rate={self._samplerate}",
            f"--demuxer-rawaudio-channels={self._channels}",
            "--demuxer-rawaudio-format=s16le",
            "-",  # Read from stdin
        ]

        if self._device:
            cmd.insert(1, f"--audio-device={self._device}")

        _LOGGER.debug("Starting mpv: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop mpv process."""
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            except Exception as e:
                _LOGGER.warning("Error closing mpv: %s", e)
            finally:
                self._proc = None

    def write(self, data: bytes) -> None:
        """Write PCM audio data to mpv.

        Args:
            data: Raw PCM audio (signed 16-bit LE)
        """
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
                self._bytes_written += len(data)
                # Log first few writes and periodically
                if self._bytes_written <= 10000 or self._bytes_written % 500000 == 0:
                    _LOGGER.debug("mpv write: %d bytes (total: %d)", len(data), self._bytes_written)
            except BrokenPipeError:
                _LOGGER.warning("mpv pipe broken - process may have exited")

    @property
    def active(self) -> bool:
        """Check if mpv process is still running."""
        return self._proc is not None and self._proc.poll() is None


@dataclass
class AudioChunk:
    """Audio chunk with timestamp for synchronized playback."""
    timestamp_us: int  # Server clock time when first sample should play
    pcm_data: bytes  # Decoded PCM audio data


class SyncState:
    """Synchronization state constants."""
    SYNCHRONIZED = "synchronized"
    SYNCING = "syncing"
    DESYNCED = "desynced"
    ERROR = "error"


class SendspinAudioPlayer:
    """Synchronized audio player for Sendspin streams.

    This player receives timestamped audio chunks, buffers them, and
    outputs them at the correct time according to the synchronized clock.
    It handles jitter absorption, late chunk detection, and resampling
    to maintain synchronization.

    Supports two audio backends:
    - mpv: Preferred for PulseAudio setups, especially in containers
    - sounddevice: Fallback using PortAudio
    """

    # Maximum time a chunk can be late before it's dropped (microseconds)
    MAX_LATE_US = 50_000  # 50ms

    # Maximum time to wait for early chunks (microseconds)
    # Server sends chunks 3-4 seconds ahead for multi-room sync buffering
    MAX_EARLY_US = 10_000_000  # 10 seconds

    # Maximum single sleep duration when waiting for chunks (microseconds)
    MAX_WAIT_SLEEP_US = 100_000  # 100ms - stay responsive while waiting

    # Minimum buffer level before starting playback (microseconds)
    MIN_BUFFER_US = 100_000  # 100ms

    def __init__(
        self,
        clock_sync: KalmanClockSync,
        output_device: Optional[str] = None,
        mpv_audio_device: Optional[str] = None,
        buffer_capacity_us: int = 2_000_000,
        on_state_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize the audio player.

        Args:
            clock_sync: Clock synchronization object
            output_device: Audio output device for sounddevice (None for default)
            mpv_audio_device: Audio device for mpv (e.g., "pulse/bluez_output.xxx").
                              If set, mpv is used instead of sounddevice.
            buffer_capacity_us: Maximum buffer size in microseconds
            on_state_change: Callback when sync state changes
        """
        # Determine which backend to use
        self._use_mpv = mpv_audio_device is not None and MPV_AVAILABLE
        self._mpv_audio_device = mpv_audio_device

        if not self._use_mpv and not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "SendspinAudioPlayer requires either mpv or sounddevice. "
                "Install mpv or: pip install sounddevice"
            )

        if self._use_mpv:
            _LOGGER.info("Sendspin will use mpv for audio output: %s", mpv_audio_device)
        else:
            _LOGGER.info("Sendspin will use sounddevice for audio output")

        self._clock_sync = clock_sync
        self._output_device = output_device
        self._buffer_capacity_us = buffer_capacity_us
        self._on_state_change = on_state_change

        # Audio configuration
        self._codec: str = "opus"
        self._sample_rate: int = 48000
        self._channels: int = 2
        self._bit_depth: int = 16
        self._decoder: Optional[AudioDecoder] = None

        # Audio buffer (thread-safe deque)
        self._buffer: Deque[AudioChunk] = deque()
        self._buffer_lock = threading.Lock()
        self._buffer_duration_us: int = 0

        # Playback state
        self._playing = threading.Event()
        self._stream: Optional["sd.OutputStream"] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._sync_state: str = SyncState.SYNCING
        self._stop_requested = threading.Event()

        # Volume control (0-100 perceived loudness)
        self._volume: int = 100
        self._muted: bool = False

        # Statistics
        self._chunks_received: int = 0
        self._chunks_played: int = 0
        self._chunks_dropped: int = 0

    def configure_stream(
        self,
        codec: str,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: Optional[bytes] = None,
    ) -> None:
        """Configure the audio stream parameters.

        Args:
            codec: Audio codec ("opus", "pcm", "flac")
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            bit_depth: Bits per sample
            codec_header: Optional codec-specific header
        """
        self._codec = codec
        self._sample_rate = sample_rate
        self._channels = channels
        self._bit_depth = bit_depth

        # Create decoder
        self._decoder = get_decoder(codec)
        self._decoder.configure(sample_rate, channels, bit_depth, codec_header)

        _LOGGER.info(
            "Sendspin stream configured: %s, %d Hz, %d ch, %d-bit",
            codec,
            sample_rate,
            channels,
            bit_depth,
        )

    def receive_chunk(self, timestamp_us: int, data: bytes) -> None:
        """Receive an audio chunk from the network.

        Args:
            timestamp_us: Server clock timestamp for playback
            data: Encoded audio data
        """
        if self._decoder is None:
            _LOGGER.warning("Received chunk but decoder not configured")
            return

        self._chunks_received += 1

        # Decode audio
        try:
            pcm_data = self._decoder.decode(data)
        except Exception as e:
            _LOGGER.warning("Failed to decode audio chunk: %s", e)
            return

        # Calculate chunk duration
        samples = len(pcm_data) // (self._channels * 2)  # 16-bit samples
        chunk_duration_us = int(samples * 1_000_000 / self._sample_rate)

        # Add to buffer
        chunk = AudioChunk(timestamp_us=timestamp_us, pcm_data=pcm_data)

        with self._buffer_lock:
            # Check if buffer is full
            if self._buffer_duration_us >= self._buffer_capacity_us:
                # Drop oldest chunk
                if self._buffer:
                    old = self._buffer.popleft()
                    old_samples = len(old.pcm_data) // (self._channels * 2)
                    old_duration = int(old_samples * 1_000_000 / self._sample_rate)
                    self._buffer_duration_us -= old_duration
                    self._chunks_dropped += 1

            self._buffer.append(chunk)
            self._buffer_duration_us += chunk_duration_us

    def start_playback(self) -> None:
        """Start audio playback."""
        if self._playing.is_set():
            return

        self._stop_requested.clear()
        self._playing.set()

        # Start playback thread
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="SendspinAudioPlayback",
            daemon=True,
        )
        self._playback_thread.start()

        _LOGGER.info("Sendspin playback started")

    def stop_playback(self) -> None:
        """Stop audio playback and clear buffer."""
        self._stop_requested.set()
        self._playing.clear()

        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)
            self._playback_thread = None

        self.clear_buffer()
        self._set_sync_state(SyncState.SYNCING)

        _LOGGER.info("Sendspin playback stopped")

    def clear_buffer(self) -> None:
        """Clear the audio buffer (e.g., after a seek)."""
        with self._buffer_lock:
            self._buffer.clear()
            self._buffer_duration_us = 0

        if self._decoder:
            self._decoder.reset()

    def set_volume(self, volume: int) -> None:
        """Set volume level.

        Args:
            volume: Volume 0-100 (perceived loudness)
        """
        self._volume = max(0, min(100, volume))
        _LOGGER.debug("Sendspin volume set to %d", self._volume)

    def set_muted(self, muted: bool) -> None:
        """Set mute state.

        Args:
            muted: True to mute, False to unmute
        """
        self._muted = muted
        _LOGGER.debug("Sendspin muted: %s", muted)

    @property
    def volume(self) -> int:
        """Current volume level (0-100)."""
        return self._volume

    @property
    def muted(self) -> bool:
        """Current mute state."""
        return self._muted

    @property
    def sync_state(self) -> str:
        """Current synchronization state."""
        return self._sync_state

    @property
    def buffer_level_ms(self) -> int:
        """Current buffer level in milliseconds."""
        return self._buffer_duration_us // 1000

    @property
    def is_playing(self) -> bool:
        """Whether playback is active."""
        return self._playing.is_set()

    def get_stats(self) -> dict:
        """Get playback statistics."""
        return {
            "chunks_received": self._chunks_received,
            "chunks_played": self._chunks_played,
            "chunks_dropped": self._chunks_dropped,
            "buffer_level_ms": self.buffer_level_ms,
            "sync_state": self._sync_state,
        }

    def _set_sync_state(self, state: str) -> None:
        """Update sync state and notify callback."""
        if state != self._sync_state:
            self._sync_state = state
            if self._on_state_change:
                try:
                    self._on_state_change(state)
                except Exception as e:
                    _LOGGER.warning("State change callback error: %s", e)

    def _playback_loop(self) -> None:
        """Main playback thread loop."""
        if self._use_mpv:
            self._playback_loop_mpv()
        else:
            self._playback_loop_sounddevice()

    def _playback_loop_mpv(self) -> None:
        """Playback loop using mpv backend."""
        try:
            with MpvOutputStream(
                device=self._mpv_audio_device,
                samplerate=self._sample_rate,
                channels=self._channels,
            ) as stream:
                _LOGGER.info(
                    "Sendspin mpv audio output opened: %s",
                    self._mpv_audio_device or "default",
                )

                # Wait for minimum buffer before starting
                _LOGGER.debug("Waiting for buffer to fill (need %d ms)...", self.MIN_BUFFER_US // 1000)
                wait_count = 0
                while self._playing.is_set() and not self._stop_requested.is_set():
                    if self._buffer_duration_us >= self.MIN_BUFFER_US:
                        _LOGGER.info(
                            "Buffer ready: %d ms, clock offset=%.0f us, sync_quality=%s",
                            self._buffer_duration_us // 1000,
                            self._clock_sync.offset_us,
                            self._clock_sync.sync_quality,
                        )
                        break
                    wait_count += 1
                    if wait_count % 100 == 0:  # Log every ~1 second
                        _LOGGER.debug("Still waiting for buffer: %d ms", self._buffer_duration_us // 1000)
                    time.sleep(0.01)

                # Main playback loop
                _LOGGER.debug("Starting main playback loop")
                while self._playing.is_set() and not self._stop_requested.is_set():
                    if not stream.active:
                        _LOGGER.error("mpv process exited unexpectedly")
                        self._set_sync_state(SyncState.ERROR)
                        break
                    self._process_audio_chunk(stream)

        except Exception as e:
            _LOGGER.error("Playback error (mpv): %s", e)
            self._set_sync_state(SyncState.ERROR)

    def _playback_loop_sounddevice(self) -> None:
        """Playback loop using sounddevice backend."""
        device = self._output_device

        # Try configured device, fall back to default if it fails
        if device is not None:
            try:
                # Validate device exists before opening stream
                sd.query_devices(device)
            except (ValueError, sd.PortAudioError) as e:
                _LOGGER.warning(
                    "Configured output device '%s' not found (%s), using system default",
                    device,
                    e,
                )
                device = None

        try:
            # Open audio stream
            with sd.OutputStream(
                device=device,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                blocksize=1024,
            ) as stream:
                self._stream = stream
                _LOGGER.info(
                    "Sendspin sounddevice output opened: %s",
                    stream.device if hasattr(stream, 'device') else "default device",
                )

                # Wait for minimum buffer before starting
                while self._playing.is_set() and not self._stop_requested.is_set():
                    if self._buffer_duration_us >= self.MIN_BUFFER_US:
                        break
                    time.sleep(0.01)

                # Main playback loop
                while self._playing.is_set() and not self._stop_requested.is_set():
                    self._process_audio_chunk(stream)

        except Exception as e:
            _LOGGER.error("Playback error (sounddevice): %s", e)
            self._set_sync_state(SyncState.ERROR)
        finally:
            self._stream = None

    def _process_audio_chunk(self, stream: Union["sd.OutputStream", MpvOutputStream]) -> None:
        """Process and play the next audio chunk if ready."""
        # Get next chunk
        chunk = self._peek_next_chunk()
        if chunk is None:
            # Buffer empty, wait a bit
            time.sleep(0.001)
            return

        # Convert server timestamp to local time
        target_time_us = self._clock_sync.server_to_local(chunk.timestamp_us)
        current_time_us = self._clock_sync.get_local_time_us()

        # Calculate timing
        delta_us = target_time_us - current_time_us

        # Track wait iterations for this chunk (reset when we play or drop)
        if not hasattr(self, '_wait_iterations'):
            self._wait_iterations = 0
            self._last_chunk_ts = 0

        # Reset counter if we're looking at a new chunk
        if chunk.timestamp_us != self._last_chunk_ts:
            self._wait_iterations = 0
            self._last_chunk_ts = chunk.timestamp_us

        self._wait_iterations += 1

        # Log periodically during wait (every 10 iterations = ~1 second)
        if self._wait_iterations == 1 or self._wait_iterations % 10 == 0:
            _LOGGER.debug(
                "Chunk wait #%d: delta=%.1f ms (%.2f s), buffer=%d ms, chunks_played=%d",
                self._wait_iterations,
                delta_us / 1000,
                delta_us / 1_000_000,
                self._buffer_duration_us // 1000,
                self._chunks_played,
            )

        if delta_us > self.MAX_EARLY_US:
            # Way too early - this indicates a severe sync issue
            _LOGGER.warning(
                "Chunk extremely early: delta=%.1f s > MAX_EARLY=%.1f s - possible clock sync issue",
                delta_us / 1_000_000, self.MAX_EARLY_US / 1_000_000
            )
            time.sleep(0.1)
            return

        if delta_us > 0:
            # Chunk is early, wait for it (in small increments to stay responsive)
            wait_us = min(delta_us, self.MAX_WAIT_SLEEP_US)
            wait_seconds = wait_us / 1_000_000
            time.sleep(wait_seconds)

            # If we haven't waited the full delta, return and re-check
            # This keeps the thread responsive during long waits
            if delta_us > self.MAX_WAIT_SLEEP_US:
                return

            self._set_sync_state(SyncState.SYNCHRONIZED)

        elif delta_us < -self.MAX_LATE_US:
            # Too late, drop this chunk
            self._pop_chunk()
            self._chunks_dropped += 1
            self._set_sync_state(SyncState.DESYNCED)
            if self._chunks_dropped < 10 or self._chunks_dropped % 100 == 0:
                _LOGGER.warning(
                    "Dropped late chunk #%d: %d us late (%.1f ms)",
                    self._chunks_dropped, -delta_us, -delta_us / 1000
                )
            return

        else:
            # On time or slightly late, play it
            self._set_sync_state(SyncState.SYNCHRONIZED)

        # Pop and play the chunk
        chunk = self._pop_chunk()
        if chunk:
            _LOGGER.info(
                "Playing chunk #%d after %d waits: delta was %.1f ms, buffer=%d ms",
                self._chunks_played + 1,
                self._wait_iterations,
                delta_us / 1000,
                self._buffer_duration_us // 1000,
            )
            self._play_audio(stream, chunk.pcm_data)
            self._chunks_played += 1
            self._wait_iterations = 0  # Reset for next chunk

    def _peek_next_chunk(self) -> Optional[AudioChunk]:
        """Peek at the next chunk without removing it."""
        with self._buffer_lock:
            if self._buffer:
                return self._buffer[0]
            return None

    def _pop_chunk(self) -> Optional[AudioChunk]:
        """Remove and return the next chunk."""
        with self._buffer_lock:
            if self._buffer:
                chunk = self._buffer.popleft()
                samples = len(chunk.pcm_data) // (self._channels * 2)
                duration_us = int(samples * 1_000_000 / self._sample_rate)
                self._buffer_duration_us -= duration_us
                return chunk
            return None

    def _play_audio(self, stream: Union["sd.OutputStream", MpvOutputStream], pcm_data: bytes) -> None:
        """Play PCM audio data with volume applied.

        Args:
            stream: Active output stream (sounddevice or mpv)
            pcm_data: Raw PCM audio (signed 16-bit LE)
        """
        if self._muted:
            # Output silence
            silence = bytes(len(pcm_data))
            stream.write(silence)
            return

        if self._volume == 100:
            # No volume adjustment needed
            stream.write(pcm_data)
            return

        # Apply volume (convert perceived loudness to amplitude)
        # Using a simple power curve for perceived loudness
        import numpy as np

        # Convert to numpy array
        samples = np.frombuffer(pcm_data, dtype=np.int16).copy()

        # Calculate amplitude multiplier from perceived volume
        # Perceived loudness is roughly proportional to amplitude^2
        # So amplitude = sqrt(perceived / 100)
        amplitude = (self._volume / 100.0) ** 0.5

        # Apply gain
        samples = (samples * amplitude).astype(np.int16)

        # Write to stream
        stream.write(samples.tobytes())
