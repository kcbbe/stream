"""
Heart-Rate Forecast Overlay Renderer

Extends VerticalLineRenderer with a secondary y-axis showing the heart-rate
feature and the live AR forecast (mean + confidence band).

Layout:
    - Left y-axis : raw ECG signal (mV), anomaly vlines (inherited behaviour)
    - Right y-axis: heart rate (bpm) staircase + forecast ahead in time

Why a twin axis:
    The forecast is in bpm while the signal is in mV; sharing one y-scale would
    squash one of them flat or push it out of view.

Why the renderer keeps its OWN beat history:
    The simulator skips appending None features during deriver warm-up, so the
    ``features`` array can be shorter than ``times`` and cannot be plotted
    against them directly. Additionally HR is a staircase (constant within a
    beat). The renderer therefore samples the LATEST feature each update and
    appends it to an internal history only when it CHANGES — the same beat
    gating used by BeatGatedARPredictor — yielding (time, bpm) points aligned
    to actual beats.

Forecast drawing:
    - ``mean``  : orange dash-dot line extending from the last observed beat
    - CI band   : light-orange fill (fill_between cannot be updated in place,
                  so it is removed and recreated each frame; negligible cost at
                  ~33 frames/sec with blit=False)

Important Dependencies:
    - streamsim.src.renderers.vline.VerticalLineRenderer: parent class
    - streamsim.src.core.interfaces.StreamingRenderer: base interface

Author: F.Feenstra

Example:
    >>> renderer = HeartRateForecastRenderer(
    ...     line_color='black',
    ...     vline_color='red',
    ...     hr_color='steelblue',
    ...     forecast_color='orange',
    ...     title_template='Heart Rate: {feature:.1f} bpm'
    ... )
"""

import logging
from typing import List, Any, Optional, Dict

import numpy as np

from streamsim.src.renderers.vline import VerticalLineRenderer

logger = logging.getLogger(__name__)


class HeartRateForecastRenderer(VerticalLineRenderer):
    """
    ECG + anomaly vlines (inherited) plus a right-axis heart-rate trace with
    live AR forecast overlay.
    """

    def __init__(
        self,
        # HR trace styling (right axis)
        hr_color: str = 'steelblue',
        hr_width: float = 1.5,
        hr_label: str = 'Heart Rate (bpm)',
        # Forecast styling (right axis)
        forecast_color: str = 'darkorange',
        forecast_style: str = '-.',
        forecast_width: float = 1.5,
        forecast_label: str = 'HR Forecast',
        band_alpha: float = 0.15,
        # Right-axis range; None = autoscale to observed HR
        hr_ylim: Optional[tuple] = None,
        # History control
        max_hr_history: int = 5000,
        **kwargs,                     # forwarded to VerticalLineRenderer
    ):
        super().__init__(**kwargs)

        self.hr_color = hr_color
        self.hr_width = hr_width
        self.hr_label = hr_label
        self.forecast_color = forecast_color
        self.forecast_style = forecast_style
        self.forecast_width = forecast_width
        self.forecast_label = forecast_label
        self.band_alpha = band_alpha
        self.hr_ylim = hr_ylim

        # Internal beat-gated HR history: (timestamp, bpm) tuples
        self._hr_history: List[tuple] = []   # plain list, trimmed manually
        self._max_hr_history = max_hr_history
        self._last_hr: Optional[float] = None

        # Artists (created in initialize)
        self.ax2 = None                # twin y-axis (shares x with main ax)
        self.hr_line = None
        self.hr_marker = None         # dots at observed beats
        self.forecast_line = None
        self.forecast_band = None     # recreated each frame (fill_between)

    # ------------------------------------------------------------------ #

    def initialize(self, ax: Any) -> List[Any]:
        """Create inherited artists on ax, then the HR/forecast artists on a
        twin axis sharing the same x-axis."""
        artists = super().initialize(ax)

        # Twin axis: shares x (and the simulator's x-limit updates) with ax
        self.ax2 = ax.twinx()
        self.ax2.set_ylabel(self.hr_label)
        if self.hr_ylim is not None:
            self.ax2.set_ylim(*self.hr_ylim)

        self.hr_line, = self.ax2.plot(
            [], [], lw=self.hr_width, color=self.hr_color,
            drawstyle='steps-post', label=self.hr_label,
        )
        self.hr_marker, = self.ax2.plot(
            [], [], '.', ms=4, color=self.hr_color,
        )
        self.forecast_line, = self.ax2.plot(
            [], [], lw=self.forecast_width, color=self.forecast_color,
            linestyle=self.forecast_style, label=self.forecast_label,
        )
        # Placeholder band; replaced in update()
        self.forecast_band = self.ax2.fill_between(
            [], [], [], color=self.forecast_color, alpha=self.band_alpha,
        )

        # Combined legend on the main axis
        if self.show_legend:
            handles_main, labels_main = ax.get_legend_handles_labels()
            handles_2, labels_2 = self.ax2.get_legend_handles_labels()
            ax.legend(handles_main + handles_2, labels_main + labels_2,
                     loc='upper left', fontsize=8)

        self.artists = artists + [self.hr_line, self.hr_marker,
                                   self.forecast_line, self.forecast_band]
        return self.artists

    # ------------------------------------------------------------------ #

    def update(
        self,
        times: np.ndarray,
        samples: np.ndarray,
        features: np.ndarray,
        change_points: np.ndarray,
        window_duration_sec: float,
        forecast: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Any]:
        """Update inherited signal/anomaly art, then the HR trace and forecast.

        Note: forecast["time"] is absolute seconds (predictor anchors at the
        last beat timestamp), so it maps directly onto the shared x-axis.
        """
        # 1. Inherited behaviour: ECG line + anomaly vlines + title
        artists = super().update(times, samples, features, change_points,
                                 window_duration_sec)

        if len(times) == 0:
            return artists

        # 2. Absorb the latest feature into beat-gated history
        latest_feature = features[-1] if len(features) > 0 else None
        if latest_feature is not None:
            if latest_feature != self._last_hr:
                self._last_hr = float(latest_feature)
                self._hr_history.append((float(times[-1]), self._last_hr))
                if len(self._hr_history) > self._max_hr_history:
                    self._hr_history = self._hr_history[-self._max_hr_history:]

        # 3. Draw HR history (points after the window start)
        if self._hr_history:
            window_start = times[-1] - window_duration_sec \
                - 30.0   # generous margin: keep forecast's future range visible
            pts = [(t, v) for (t, v) in self._hr_history if t >= window_start]
            if pts:
                ht = np.array([p[0] for p in pts])
                hv = np.array([p[1] for p in pts])
                self.hr_line.set_data(ht, hv)
                self.hr_marker.set_data(ht, hv)
                self.hr_line.set_visible(True)
                self.hr_marker.set_visible(True)
            else:
                self.hr_line.set_visible(False)
                self.hr_marker.set_visible(False)

            # Autoscale right axis gently to observed range (+10% headroom)
            if self.hr_ylim is None and len(self._hr_history) > 5:
                vals = np.array([v for _, v in self._hr_history])
                lo, hi = vals.min(), vals.max()
                pad = 0.1 * max(hi - lo, 1.0)
                self.ax2.set_ylim(lo - pad, hi + pad)

        # 4. Draw the forecast, if one is available
        if forecast is not None and len(forecast.get("mean", [])) > 0:
            ft = np.asarray(forecast["time"], dtype=float)
            fm = np.asarray(forecast["mean"], dtype=float)

            # Anchor visual continuity: connect from last observed beat
            if self._hr_history:
                last_t, last_v = self._hr_history[-1]
                ft = np.concatenate(([last_t], ft))
                fm = np.concatenate(([last_v], fm))

            self.forecast_line.set_data(ft, fm)
            self.forecast_line.set_visible(True)

            # fill_between has no set_data; replace the polygon each frame
            if self.forecast_band is not None:
                try:
                    self.forecast_band.remove()
                except Exception:
                    pass
            fl = np.asarray(forecast["lower"], dtype=float)
            fu = np.asarray(forecast["upper"], dtype=float)
            if self._hr_history:
                last_t, last_v = self._hr_history[-1]
                fl = np.concatenate(([last_v], fl))
                fu = np.concatenate(([last_v], fu))
            self.forecast_band = self.ax2.fill_between(
                ft, fl, fu,
                color=self.forecast_color, alpha=self.band_alpha,
            )
        else:
            self.forecast_line.set_visible(False)
            if self.forecast_band is not None:
                try:
                    self.forecast_band.remove()
                except Exception:
                    pass
                self.forecast_band = None

        return artists + [self.hr_line, self.hr_marker,
                          self.forecast_line] + \
            ([self.forecast_band] if self.forecast_band is not None else [])

    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Release inherited artists, then HR/forecast artists and history."""
        super().cleanup()
        self._hr_history = []
        self._last_hr = None
        self.ax2 = None
        self.hr_line = None
        self.hr_marker = None
        self.forecast_line = None
        self.forecast_band = None
