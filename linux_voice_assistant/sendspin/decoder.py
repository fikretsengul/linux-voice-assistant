"""Audio codec decoders for Sendspin Protocol.

This module provides audio decoders for the codecs supported by Sendspin:
- Opus (primary, low-latency)
- FLAC (lossless)
- PCM (raw, passthrough)

The opuslib dependency is optional; if not installed, Opus decoding
will raise an error when attempted.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Try to import opuslib, but don't fail if not available
try:
    import opuslib
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False
    _LOGGER.warning("opuslib not installed - Opus codec will not be available")


class AudioDecoder(ABC):
    """Abstract base class for audio decoders."""

    @abstractmethod
    def configure(
        self,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: Optional[bytes] = None,
    ) -> None:
        """Configure the decoder with stream parameters.

        Args:
            sample_rate: Sample rate in Hz (e.g., 48000)
            channels: Number of audio channels (e.g., 2 for stereo)
            bit_depth: Bits per sample (e.g., 16)
            codec_header: Optional codec-specific header (e.g., FLAC streaminfo)
        """
        pass

    @abstractmethod
    def decode(self, data: bytes) -> bytes:
        """Decode an encoded audio frame to PCM.

        Args:
            data: Encoded audio data

        Returns:
            Raw PCM audio data (signed 16-bit little-endian by default)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset decoder state (e.g., after a seek)."""
        pass


class OpusDecoder(AudioDecoder):
    """Opus audio decoder using opuslib.

    Opus is the preferred codec for Sendspin due to its low latency
    and excellent quality at various bitrates.
    """

    def __init__(self) -> None:
        if not OPUS_AVAILABLE:
            raise RuntimeError(
                "Opus decoder requires opuslib. "
                "Install with: pip install opuslib"
            )
        self._decoder: Optional["opuslib.Decoder"] = None
        self._sample_rate: int = 48000
        self._channels: int = 2
        self._frame_size: int = 960  # 20ms at 48kHz

    def configure(
        self,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: Optional[bytes] = None,
    ) -> None:
        """Configure the Opus decoder.

        Args:
            sample_rate: Must be 8000, 12000, 16000, 24000, or 48000
            channels: Must be 1 or 2
            bit_depth: Ignored for Opus (always decodes to 16-bit)
            codec_header: Ignored for Opus
        """
        self._sample_rate = sample_rate
        self._channels = channels

        # Calculate frame size (Opus uses 20ms frames typically)
        self._frame_size = sample_rate // 50  # 20ms

        self._decoder = opuslib.Decoder(sample_rate, channels)
        _LOGGER.info(
            "Opus decoder configured: %d Hz, %d channels",
            sample_rate,
            channels,
        )

    def decode(self, data: bytes) -> bytes:
        """Decode an Opus frame to PCM.

        Args:
            data: Opus-encoded frame

        Returns:
            Raw PCM audio (signed 16-bit little-endian interleaved)
        """
        if self._decoder is None:
            raise RuntimeError("Decoder not configured")

        # Opus decoder returns bytes (signed 16-bit LE)
        pcm = self._decoder.decode(data, self._frame_size)
        return pcm

    def reset(self) -> None:
        """Reset the Opus decoder state."""
        # Recreate the decoder to reset state
        if self._decoder is not None:
            self._decoder = opuslib.Decoder(self._sample_rate, self._channels)


class PCMDecoder(AudioDecoder):
    """Passthrough decoder for raw PCM audio.

    Simply passes through the raw PCM data with optional format conversion.
    """

    def __init__(self) -> None:
        self._sample_rate: int = 48000
        self._channels: int = 2
        self._bit_depth: int = 16
        self._source_bit_depth: int = 16

    def configure(
        self,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: Optional[bytes] = None,
    ) -> None:
        """Configure PCM passthrough.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of channels
            bit_depth: Bits per sample (16 or 24)
            codec_header: Ignored
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._source_bit_depth = bit_depth
        _LOGGER.info(
            "PCM decoder configured: %d Hz, %d channels, %d-bit",
            sample_rate,
            channels,
            bit_depth,
        )

    def decode(self, data: bytes) -> bytes:
        """Pass through PCM data with optional bit depth conversion.

        Args:
            data: Raw PCM audio

        Returns:
            PCM audio (converted to 16-bit if needed)
        """
        if self._source_bit_depth == 16:
            # No conversion needed
            return data
        elif self._source_bit_depth == 24:
            # Convert 24-bit to 16-bit
            return self._convert_24_to_16(data)
        else:
            _LOGGER.warning(
                "Unsupported bit depth %d, passing through",
                self._source_bit_depth,
            )
            return data

    def _convert_24_to_16(self, data: bytes) -> bytes:
        """Convert 24-bit PCM to 16-bit PCM.

        Args:
            data: 24-bit PCM audio (3 bytes per sample, little-endian)

        Returns:
            16-bit PCM audio (2 bytes per sample, little-endian)
        """
        num_samples = len(data) // 3
        output = bytearray(num_samples * 2)

        for i in range(num_samples):
            # Read 24-bit sample (little-endian, signed)
            b0 = data[i * 3]
            b1 = data[i * 3 + 1]
            b2 = data[i * 3 + 2]

            # Convert to signed 32-bit
            sample = b0 | (b1 << 8) | (b2 << 16)
            if sample & 0x800000:
                sample |= 0xFF000000  # Sign extend

            # Scale to 16-bit
            sample16 = sample >> 8

            # Clamp
            sample16 = max(-32768, min(32767, sample16))

            # Write little-endian
            output[i * 2] = sample16 & 0xFF
            output[i * 2 + 1] = (sample16 >> 8) & 0xFF

        return bytes(output)

    def reset(self) -> None:
        """Reset decoder state (no-op for PCM)."""
        pass


class FLACDecoder(AudioDecoder):
    """FLAC audio decoder.

    Uses soundfile for FLAC decoding. This is a heavier dependency
    than Opus but provides lossless audio.
    """

    def __init__(self) -> None:
        self._sample_rate: int = 48000
        self._channels: int = 2
        self._bit_depth: int = 16
        self._codec_header: Optional[bytes] = None

    def configure(
        self,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header: Optional[bytes] = None,
    ) -> None:
        """Configure FLAC decoder.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of channels
            bit_depth: Bits per sample
            codec_header: FLAC streaminfo header (required)
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._bit_depth = bit_depth
        self._codec_header = codec_header
        _LOGGER.info(
            "FLAC decoder configured: %d Hz, %d channels, %d-bit",
            sample_rate,
            channels,
            bit_depth,
        )

    def decode(self, data: bytes) -> bytes:
        """Decode FLAC frame to PCM.

        Note: FLAC frame decoding is complex and typically requires
        full file context. This is a simplified implementation.

        Args:
            data: FLAC-encoded frame

        Returns:
            Raw PCM audio
        """
        try:
            import soundfile as sf
            import io

            # Create complete FLAC data with header
            if self._codec_header:
                full_data = self._codec_header + data
            else:
                full_data = data

            # Decode using soundfile
            audio, sr = sf.read(io.BytesIO(full_data), dtype="int16")

            # Convert to bytes
            return audio.tobytes()

        except ImportError:
            raise RuntimeError(
                "FLAC decoder requires soundfile. "
                "Install with: pip install soundfile"
            )
        except Exception as e:
            _LOGGER.warning("FLAC decode error: %s", e)
            # Return silence on error
            return bytes(1920 * self._channels)

    def reset(self) -> None:
        """Reset decoder state."""
        pass


def get_decoder(codec: str) -> AudioDecoder:
    """Factory function to create an audio decoder.

    Args:
        codec: Codec name ("opus", "pcm", "flac")

    Returns:
        Appropriate AudioDecoder instance

    Raises:
        ValueError: If codec is not supported
    """
    codec = codec.lower()

    if codec == "opus":
        return OpusDecoder()
    elif codec == "pcm":
        return PCMDecoder()
    elif codec == "flac":
        return FLACDecoder()
    else:
        raise ValueError(f"Unsupported codec: {codec}")


def is_codec_available(codec: str) -> bool:
    """Check if a codec decoder is available.

    Args:
        codec: Codec name ("opus", "pcm", "flac")

    Returns:
        True if the codec can be decoded
    """
    codec = codec.lower()

    if codec == "opus":
        return OPUS_AVAILABLE
    elif codec == "pcm":
        return True  # Always available
    elif codec == "flac":
        try:
            import soundfile
            return True
        except ImportError:
            return False
    else:
        return False
