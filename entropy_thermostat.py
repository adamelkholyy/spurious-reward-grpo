"""Entropy thermostat operating directly on the observed training entropy.

The controller is deliberately separate from TRL so its behaviour is easy to
test and reason about.  It implements a bang-bang controller with a target
deadband:

    H < target - deadband  -> SR + entropy-up clipping
    H > target + deadband  -> SR + entropy-down clipping
    otherwise              -> ground-truth training

Entropy can be EMA-smoothed and the controller can be evaluated only every N
optimizer steps.  The latter is useful when a GRPO rollout is reused for
multiple optimizer updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThermostatObservation:
    step: int
    raw_entropy: float
    control_entropy: float
    error: float
    state: str
    action: str
    switched: bool
    epsilon_low: float | None
    epsilon_high: float | None


class EntropyThermostat:
    """Controller selecting GT training or an SR entropy-correction regime."""

    def __init__(
        self,
        *,
        target: float,
        deadband: float = 0.05,
        ema_alpha: float = 0.2,
        control_interval: int = 16,
        up_epsilon_low: float = 0.05,
        up_epsilon_high: float = math.inf,
        down_epsilon_low: float = 1.0,
        down_epsilon_high: float = 0.10,
    ) -> None:
        if target < 0:
            raise ValueError("target entropy must be >= 0")
        if deadband < 0:
            raise ValueError("deadband must be >= 0")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if control_interval < 1:
            raise ValueError("control_interval must be >= 1")
        for name, value in (
            ("up_epsilon_low", up_epsilon_low),
            ("up_epsilon_high", up_epsilon_high),
            ("down_epsilon_low", down_epsilon_low),
            ("down_epsilon_high", down_epsilon_high),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

        self.target = float(target)
        self.deadband = float(deadband)
        self.ema_alpha = float(ema_alpha)
        self.control_interval = int(control_interval)
        self.up_epsilon_low = float(up_epsilon_low)
        self.up_epsilon_high = float(up_epsilon_high)
        self.down_epsilon_low = float(down_epsilon_low)
        self.down_epsilon_high = float(down_epsilon_high)

        self.state = "gt"
        self.ema_entropy: float | None = None
        self.last_control_step: int | None = None
        self._observations_since_control = 0

    def observe(self, entropy: float, step: int) -> ThermostatObservation:
        """Consume one completed optimizer-step entropy observation.

        A decision is made after every ``control_interval`` completed optimizer
        steps.  Inside the target band the controller returns to GT training.
        """
        raw_entropy = float(entropy)
        if not math.isfinite(raw_entropy) or raw_entropy < 0:
            raise ValueError(f"entropy must be finite and >= 0, got {raw_entropy}")

        if self.ema_entropy is None:
            self.ema_entropy = raw_entropy
        else:
            a = self.ema_alpha
            self.ema_entropy = a * raw_entropy + (1.0 - a) * self.ema_entropy

        self._observations_since_control += 1
        due = self._observations_since_control >= self.control_interval
        switched = False
        action = "wait"

        if due:
            self.last_control_step = int(step)
            self._observations_since_control = 0
            lower = self.target - self.deadband
            upper = self.target + self.deadband

            if self.ema_entropy < lower:
                action = "sr_up"
                if self.state != "sr_up":
                    self.state = "sr_up"
                    switched = True
            elif self.ema_entropy > upper:
                action = "sr_down"
                if self.state != "sr_down":
                    self.state = "sr_down"
                    switched = True
            else:
                action = "gt"
                if self.state != "gt":
                    self.state = "gt"
                    switched = True

        epsilon_low, epsilon_high = self.current_epsilons()
        return ThermostatObservation(
            step=int(step),
            raw_entropy=raw_entropy,
            control_entropy=self.ema_entropy,
            error=self.ema_entropy - self.target,
            state=self.state,
            action=action,
            switched=switched,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
        )

    def current_epsilons(self) -> tuple[float | None, float | None]:
        """Return clipping for the current state; ``None`` means unchanged."""
        if self.state == "sr_up":
            return self.up_epsilon_low, self.up_epsilon_high
        if self.state == "sr_down":
            return self.down_epsilon_low, self.down_epsilon_high
        return None, None
