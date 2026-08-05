"""Small, dependency-free controller for target-entropy GRPO experiments.

The controller is deliberately separate from TRL so its behaviour is easy to
test and reason about.  It implements a bang-bang thermostat with hysteresis:

    H < target - deadband  -> entropy-up clipping
    H > target + deadband  -> entropy-down clipping
    otherwise              -> keep the current regime

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
    """Hysteretic controller that selects entropy-up/down clipping regimes."""

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

        self.state = "initial"
        self.ema_entropy: float | None = None
        self.last_control_step: int | None = None
        self._observations_since_control = 0

    def observe(self, entropy: float, step: int) -> ThermostatObservation:
        """Consume one completed optimizer-step entropy observation.

        A decision is made after every ``control_interval`` completed optimizer
        steps.  Inside the deadband the current clipping regime is retained
        (true hysteresis).
        """
        entropy = float(entropy)
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(f"entropy must be finite and >= 0, got {entropy}")

        if self.ema_entropy is None:
            self.ema_entropy = entropy
        else:
            a = self.ema_alpha
            self.ema_entropy = a * entropy + (1.0 - a) * self.ema_entropy

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
                action = "up"
                if self.state != "up":
                    self.state = "up"
                    switched = True
            elif self.ema_entropy > upper:
                action = "down"
                if self.state != "down":
                    self.state = "down"
                    switched = True
            else:
                # Do not switch inside the band: keeping the previous state is
                # what gives the controller hysteresis and prevents chatter.
                action = "hold"

        epsilon_low, epsilon_high = self.current_epsilons()
        return ThermostatObservation(
            step=int(step),
            raw_entropy=entropy,
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
        if self.state == "up":
            return self.up_epsilon_low, self.up_epsilon_high
        if self.state == "down":
            return self.down_epsilon_low, self.down_epsilon_high
        return None, None
