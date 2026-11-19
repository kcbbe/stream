"""
Streaming AR-based Feature Predictor

Provides online autoregressive (AR) forecasting of derived streaming features,
designed to plug into the StreamingSimulator's producer-consumer architecture.

Model:
    f_t = c + phi_1 * f_{t-1} + phi_2 * f_{t-2} + ... + phi_p * f_{t-p} + eps_t

The model is fitted by ordinary least squares on a sliding window of the most
recent feature values and refreshed every ``refit_every`` samples, so the
forecast adapts continuously to the evolving stream.

Forecasts:
    - ``mean``: recursive multi-step-ahead point forecast
    - ``lower`` / ``upper``: approximate confidence bands whose width grows
      with horizon, derived from the in-window residual variance and the AR
      psi-weights (impulse response), i.e. Var[h-step] = sigma^2 * sum(psi_k^2)

Change point integration:
    ``on_change_point()`` may be called when the StreamingChangePointDetector
    fires; it drops the older half of the history buffer so that stale
    pre-change data stops contaminating fits sooner, then refits.

Threading model:
    Like the feature deriver and change point detector, all methods are
    intended to be called only from the processing thread. No locking needed.

Usage:
    >>> predictor = StreamingARPredictor(order=8, horizon=50)
    >>> sim = StreamingSimulator(..., predictor=predictor)

Dependencies:
    - numpy.linalg.lstsq: least-squares AR fitting
    - collections.deque: bounded history buffers
    - streamsim.src.core.interfaces.StreamingPredictor: pluggable interface


"""

import logging
from collections import deque
from typing import Optional, Dict

import numpy as np

from streamsim.src.core.interfaces import StreamingPredictor

logger = logging.getLogger(__name__)




class StreamingARPredictor(StreamingPredictor):
    """
    Online autoregressive predictor for streaming feature values.

    Maintains a bounded history of derived feature values, periodically refits
    an AR(p) model over that window, and produces a multi-step-ahead forecast
    with approximate confidence bands.

    Args:
        order (int): AR order p (number of lags).
        window_size (int): Maximum number of recent feature values used for
            fitting. Larger windows give more stable coefficients but react
            slower to regime changes.
        horizon (int): Number of steps ahead to forecast.
        refit_every (int): Refit the model every this many added samples.
        ci_z (float): Z-score for the confidence band (1.96 -> ~95%).
    """

    def __init__(
        self,
        order: int = 8,
        window_size: int = 300,
        horizon: int = 50,
        refit_every: int = 5,
        ci_z: float = 1.96,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if window_size < order + 2:
            raise ValueError("window_size must be at least order + 2")

        self.order = order
        self.window_size = window_size
        self.horizon = horizon
        self.refit_every = max(1, refit_every)
        self.ci_z = ci_z

        self._values = deque(maxlen=window_size)       # f_{t-n} .. f_{t-1}
        self._times = deque(maxlen=window_size)        # matching timestamps
        self._samples_since_fit = 0

        # Fitted model state (None until first successful fit)
        self._phi: Optional[np.ndarray] = None   # phi[j-1] weights f_{t-j}
        self._intercept: float = 0.0
        self._resid_var: float = 0.0

    # ------------------------------------------------------------------ #
    # StreamingPredictor interface
    # ------------------------------------------------------------------ #

    def add_feature(self, feature_value: Optional[float],
                    timestamp: Optional[float] = None) -> None:
        """
        Feed the latest derived feature into the predictor.

        Ignores None values (insufficient data from the deriver) and triggers
        a periodic refit every ``refit_every`` accepted samples.
        """
        if feature_value is None:
            return

        try:
            value = float(feature_value)
        except (TypeError, ValueError):
            logger.warning("Non-numeric feature value ignored: %r", feature_value)
            return

        if not np.isfinite(value):
            logger.warning("Non-finite feature value ignored: %r", feature_value)
            return

        self._values.append(value)
        if timestamp is not None:
            self._times.append(float(timestamp))

        self._samples_since_fit += 1
        if self._samples_since_fit >= self.refit_every:
            self._samples_since_fit = 0
            self._fit()

    def get_forecast(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Produce the current multi-step forecast.

        Returns:
            dict with keys:
                - ``time``  : future timestamps for each horizon step
                - ``mean``  : point forecast per step
                - ``lower`` : lower confidence bound per step
                - ``upper`` : upper confidence bound per step
                - ``dt``    : estimated sampling period used for time axis
            or None during warm-up (not enough data / no fit yet).
        """
        if self._phi is None or len(self._values) <= self.order:
            return None

        mean = self._forecast_mean()
        std = self._forecast_std()

        dt = self._estimate_dt()
        t0 = self._times[-1] if self._times else float(len(self._values))
        future_times = t0 + dt * np.arange(1, self.horizon + 1)

        return {
            "time": future_times,
            "mean": mean,
            "lower": mean - self.ci_z * std,
            "upper": mean + self.ci_z * std,
            "dt": dt,
        }

    def on_change_point(self) -> None:
        """
        React to a detected change point.

        Drops the oldest half of the history buffer (keeping at least the
        minimum needed to fit) so post-change data dominates the next refit
        sooner, then attempts an immediate refit on what remains.
        """
        if len(self._values) <= self.order + 2:
            return

        keep = max(self.order + 2, len(self._values) // 2)
        recent = list(self._values)[-keep:]
        self._values.clear()
        self._values.extend(recent)

        logger.debug(
            "[ARPredictor] Change point: truncated history to %d samples, refitting",
            keep,
        )
        self._fit()

    def reset(self) -> None:
        """Clear all history and fitted model state."""
        self._values.clear()
        self._times.clear()
        self._samples_since_fit = 0
        self._phi = None
        self._intercept = 0.0
        self._resid_var = 0.0

    # ------------------------------------------------------------------ #
    # Internal machinery
    # ------------------------------------------------------------------ #

    def _fit(self) -> None:
        """
        Fit AR(p) by least squares over the current history window.

        Builds the standard lag design matrix via sliding windows:
            row i = [f_i, f_{i+1}, ..., f_{i+p-1}, 1], target = f_{i+p}
        so ``coef[:p]`` aligns with lags [f_{t-p}, ..., f_{t-1}] (oldest to
        newest) and is reversed into ``phi`` (phi[0] weights f_{t-1}).
        Residual variance (dof-corrected) drives the confidence bands.
        """
        data = np.fromiter(self._values, dtype=float, count=len(self._values))
        n = data.size
        if n < self.order + 2:
            return

        # Windows [f_i .. f_{i+p-1}]; last window has no successor target.
        X = np.lib.stride_tricks.sliding_window_view(data, self.order)[:-1]
        y = data[self.order:]

        X = np.hstack([X, np.ones((X.shape[0], 1))])   # intercept column
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        residuals = y - X @ coef
        dof = max(1, X.shape[0] - (self.order + 1))
        self._resid_var = float(residuals @ residuals) / dof

        self._phi = np.ascontiguousarray(coef[: self.order][::-1])
        self._intercept = float(coef[-1])

        # Cheap stability diagnostic: sum of |phi| >= 1 does not prove
        # instability but is a practical warning sign for explosive paths.
        if np.abs(self._phi).sum() >= 1.0:
            logger.debug(
                "[ARPredictor] Warning: sum|phi| = %.3f, forecast may diverge",
                float(np.abs(self._phi).sum()),
            )

    def _forecast_mean(self) -> np.ndarray:
        """Recursive multi-step point forecast along the horizon."""
        window = list(self._values)[-self.order:]      # oldest .. newest
        p = self.order
        mean = np.empty(self.horizon)

        w = window  # grows as predictions are appended
        for h in range(self.horizon):
            pred = self._intercept
            for j in range(1, p + 1):
                pred += self._phi[j - 1] * w[-j]
            w = w + [pred] if isinstance(w, list) else w
            mean[h] = pred
        return mean

    def _forecast_std(self) -> np.ndarray:
        """
        Per-step forecast standard deviation.

        Var[h-step] = sigma^2 * sum_{k=0}^{h-1} psi_k^2, where psi are the AR
        impulse-response weights: psi_0 = 1, psi_k = sum_j phi_j * psi_{k-j}.
        """
        p = self.order
        sigma2 = max(self._resid_var, 1e-12)

        psi = [1.0]
        var = np.empty(self.horizon)
        var[0] = sigma2
        for h in range(1, self.horizon):
            while len(psi) <= h:
                k = len(psi)
                nxt = 0.0
                for j in range(1, min(k, p) + 1):
                    nxt += self._phi[j - 1] * psi[k - j]
                psi.append(nxt)
            var[h] = var[h - 1] + sigma2 * psi[h] ** 2

        return np.sqrt(var)

    def _estimate_dt(self) -> float:
        """
        Estimate the sampling period from recent timestamps.

        Uses the median of positive inter-sample gaps over the window, which
        is robust to occasional jitter. Falls back to 1.0 when timestamps are
        absent or degenerate (AR is index-based in that case).
        """
        if len(self._times) >= 2:
            times = np.fromiter(self._times, dtype=float, count=len(self._times))
            diffs = np.diff(times)
            positive = diffs[diffs > 0]
            if positive.size:
                return float(np.median(positive))
        return 1.0
