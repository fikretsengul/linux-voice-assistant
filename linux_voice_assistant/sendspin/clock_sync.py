"""Kalman filter-based clock synchronization for Sendspin Protocol.

This module implements a two-dimensional Kalman filter that tracks both
clock offset and drift between the client and server. This allows for
microsecond-level synchronization accuracy required for multi-room audio.

The Kalman filter approach is recommended by the Sendspin specification
for achieving stable, accurate time synchronization even with network jitter.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class ClockSyncStats:
    """Statistics about clock synchronization quality."""
    offset_us: float  # Current estimated offset in microseconds
    drift_ppm: float  # Estimated drift in parts per million
    variance: float  # Uncertainty in offset estimate
    samples: int  # Number of time sync samples processed
    last_rtt_us: int  # Last measured round-trip time


class KalmanClockSync:
    """Kalman filter for clock offset and drift estimation.

    The filter tracks two states:
    1. Clock offset (local_time - server_time) in microseconds
    2. Clock drift rate in microseconds per second

    This allows the filter to predict the offset even between measurements
    and compensate for systematic clock frequency differences.
    """

    def __init__(
        self,
        initial_offset: float = 0.0,
        initial_variance: float = 1e12,
        process_noise_offset: float = 1e6,
        process_noise_drift: float = 1e2,
        measurement_noise: float = 1e8,
    ) -> None:
        """Initialize the Kalman filter.

        Args:
            initial_offset: Starting offset estimate (microseconds)
            initial_variance: Initial uncertainty in offset
            process_noise_offset: How much offset can change per second
            process_noise_drift: How much drift rate can change per second
            measurement_noise: Expected variance in measurements (network jitter)
        """
        # State estimates
        self._offset: float = initial_offset  # Offset in microseconds
        self._drift: float = 0.0  # Drift in us/second

        # Covariance matrix (2x2, but we store as separate values)
        self._var_offset: float = initial_variance
        self._var_drift: float = 1e6  # Initial drift variance
        self._cov_offset_drift: float = 0.0

        # Process and measurement noise
        self._q_offset: float = process_noise_offset
        self._q_drift: float = process_noise_drift
        self._r: float = measurement_noise

        # Timing
        self._last_update_time: Optional[float] = None
        self._samples: int = 0
        self._last_rtt_us: int = 0

    def update(
        self,
        client_transmitted_us: int,
        server_received_us: int,
        server_transmitted_us: int,
        client_received_us: int,
    ) -> None:
        """Update the filter with a new time sync measurement.

        Uses the standard NTP-like round-trip calculation:
        - RTT = (client_received - client_transmitted) - (server_transmitted - server_received)
        - Offset = ((server_received - client_transmitted) + (server_transmitted - client_received)) / 2

        Args:
            client_transmitted_us: When client sent the time request
            server_received_us: When server received the request
            server_transmitted_us: When server sent the response
            client_received_us: When client received the response
        """
        current_time = time.monotonic()

        # Calculate RTT and measured offset
        rtt_us = (
            (client_received_us - client_transmitted_us)
            - (server_transmitted_us - server_received_us)
        )
        self._last_rtt_us = max(0, rtt_us)

        # Measured offset using symmetric assumption
        # offset = local_time - server_time (positive means local is ahead)
        measured_offset = (
            (client_transmitted_us - server_received_us)
            + (client_received_us - server_transmitted_us)
        ) / 2

        # Time since last update (for drift prediction)
        if self._last_update_time is not None:
            dt = current_time - self._last_update_time
        else:
            dt = 0.0

        # === Prediction Step ===
        if dt > 0:
            # Predict offset using drift
            predicted_offset = self._offset + self._drift * dt

            # Update covariance with process noise
            self._var_offset += (
                2 * self._cov_offset_drift * dt
                + self._var_drift * dt * dt
                + self._q_offset * dt
            )
            self._cov_offset_drift += self._var_drift * dt
            self._var_drift += self._q_drift * dt
        else:
            predicted_offset = self._offset

        # === Update Step ===
        # Innovation (measurement residual)
        innovation = measured_offset - predicted_offset

        # Measurement noise based on RTT (higher RTT = more uncertainty)
        adaptive_noise = self._r + (rtt_us ** 2) / 4

        # Innovation covariance
        s = self._var_offset + adaptive_noise

        # Kalman gains
        k_offset = self._var_offset / s
        k_drift = self._cov_offset_drift / s

        # Update state estimates
        self._offset = predicted_offset + k_offset * innovation
        self._drift = self._drift + k_drift * innovation

        # Update covariance
        self._var_offset = (1 - k_offset) * self._var_offset
        self._cov_offset_drift = (1 - k_offset) * self._cov_offset_drift
        self._var_drift = self._var_drift - k_drift * self._cov_offset_drift

        # Ensure covariance stays positive definite
        self._var_offset = max(self._var_offset, 1.0)
        self._var_drift = max(self._var_drift, 0.01)

        self._last_update_time = current_time
        self._samples += 1

        _LOGGER.debug(
            "Clock sync update: offset=%.1f us, drift=%.2f ppm, rtt=%d us",
            self._offset,
            self._drift * 1e6 / 1e6,  # Convert to ppm
            rtt_us,
        )

    def server_to_local(self, server_time_us: int) -> int:
        """Convert a server timestamp to local time.

        Args:
            server_time_us: Timestamp in server's clock (microseconds)

        Returns:
            Equivalent timestamp in local clock (microseconds)
        """
        # Predict current offset using drift
        if self._last_update_time is not None:
            dt = time.monotonic() - self._last_update_time
            current_offset = self._offset + self._drift * dt
        else:
            current_offset = self._offset

        return server_time_us + int(current_offset)

    def local_to_server(self, local_time_us: int) -> int:
        """Convert a local timestamp to server time.

        Args:
            local_time_us: Timestamp in local clock (microseconds)

        Returns:
            Equivalent timestamp in server's clock (microseconds)
        """
        # Predict current offset using drift
        if self._last_update_time is not None:
            dt = time.monotonic() - self._last_update_time
            current_offset = self._offset + self._drift * dt
        else:
            current_offset = self._offset

        return local_time_us - int(current_offset)

    def get_current_server_time(self) -> int:
        """Get the estimated current server time.

        Returns:
            Estimated current server time in microseconds
        """
        local_time_us = int(time.monotonic() * 1_000_000)
        return self.local_to_server(local_time_us)

    def get_local_time_us(self) -> int:
        """Get current local time in microseconds.

        Returns:
            Current local monotonic time in microseconds
        """
        return int(time.monotonic() * 1_000_000)

    @property
    def offset_us(self) -> float:
        """Current estimated offset in microseconds."""
        if self._last_update_time is not None:
            dt = time.monotonic() - self._last_update_time
            return self._offset + self._drift * dt
        return self._offset

    @property
    def drift_ppm(self) -> float:
        """Estimated drift rate in parts per million."""
        # drift is in us/second, convert to ppm (us/second * 1e6 / 1e6)
        return self._drift

    @property
    def variance(self) -> float:
        """Current variance (uncertainty) in offset estimate."""
        return self._var_offset

    @property
    def is_synced(self) -> bool:
        """Check if clock is sufficiently synchronized.

        Returns True if variance is below threshold and we have
        enough samples.
        """
        # Consider synced if variance < 1ms^2 and at least 3 samples
        return self._var_offset < 1e6 and self._samples >= 3

    @property
    def sync_quality(self) -> str:
        """Human-readable sync quality indicator."""
        if self._samples == 0:
            return "not_synced"
        elif self._var_offset > 1e9:
            return "poor"
        elif self._var_offset > 1e6:
            return "fair"
        else:
            return "good"

    def get_stats(self) -> ClockSyncStats:
        """Get current synchronization statistics."""
        return ClockSyncStats(
            offset_us=self.offset_us,
            drift_ppm=self.drift_ppm,
            variance=self.variance,
            samples=self._samples,
            last_rtt_us=self._last_rtt_us,
        )

    def reset(self) -> None:
        """Reset the filter to initial state."""
        self._offset = 0.0
        self._drift = 0.0
        self._var_offset = 1e12
        self._var_drift = 1e6
        self._cov_offset_drift = 0.0
        self._last_update_time = None
        self._samples = 0
        self._last_rtt_us = 0
