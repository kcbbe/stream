"""
ECG Streaming Predictor Demo

Demonstrates the StreamingARPredictor on the synthetic ECG pipeline in two
selectable modes:

HR mode (default)
    Runs the full existing pipeline (HRFeatureDeriver -> HeartRateAnomalyDetector
    -> VerticalLineRenderer) and adds a beat-gated AR forecaster of heart rate.

    Beat gating: get_feature() holds the last smoothed HR constant for ~360
    samples (one beat at 360 Hz). A naive predictor would ingest ~360 duplicate
    values per beat, turning the AR model into a persistence predictor. The
    BeatGatedARPredictor subclass below appends a value to the model history
    only when the feature CHANGES, yielding:
        - one model update per heartbeat
        - dt estimate == the average RR interval, so forecast["time"] is
          expressed in true seconds ahead
        - forecast horizon counted in BEATS, not samples

    The VerticalLineRenderer ignores the forecast kwarg (safe with the patched
    signature), so predictions are surfaced via the log and the console banner.

RAW mode (optional)
    Swaps in a pass-through deriver so the AR model forecasts the raw ECG
    waveform itself. With order spanning ~1 cardiac cycle the multi-step
    forecast reproduces the PQRST shape ahead of the live trace. Uses a
    no-op detector: the HR anomaly detector must not receive raw samples.

Verified simulator wiring (simulator (2).py):
    - StreamingSimulator(..., predictor=predictor)
    - _processing_loop calls add_feature() when feature is not None
    - on_change_point() fires inside the confirmed-change branch
    - forecast travels the queue and reaches renderer.update(..., forecast)

Requires:
    - streamsim/src/core/ar_predictor.py (StreamingARPredictor)
    - streamsim/src/core/interfaces.py with StreamingPredictor ABC
    - patched renderer signatures (forecast kwarg)

Author: F.Feenstra
"""

import logging
from typing import Optional

from streamsim.src.core.config import LoggingSetup
from streamsim.src.core.config import LOG_DIR
from streamsim.src.core.interfaces import (
    StreamingFeatureDeriver,
    StreamingChangePointDetector,
    StreamingPredictor
)
from datetime import datetime

# This MUST happen before other framework imports configure loggers
logging_setup = LoggingSetup(
    level=logging.DEBUG,
    log_file=f"{LOG_DIR}/ecg_predictor_demo{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    timestamp_filename=True,
)
logging_setup.setup_logging()
logger = logging.getLogger(__name__)

import numpy as np
from streamsim.src.core.simulator import StreamingSimulator
from streamsim.src.core.config import PlottingSetup
from streamsim.src.predictors.ar_predictor import StreamingARPredictor
from streamsim.src.features.heart_rate import HRFeatureDeriver
from streamsim.src.detectors.hr_anomaly import HeartRateAnomalyDetector
#from streamsim.src.renderers.vline import VerticalLineRenderer
from streamsim.src.renderers.hr_forecast import HeartRateForecastRenderer
from streamsim.src.sources.ecg import (
    create_ecg_with_temporary_anomaly,
    create_ecg_data_source,
)
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Helper components
# ---------------------------------------------------------------------------

class BeatGatedARPredictor(StreamingARPredictor):
    """
    StreamingARPredictor that ingests a feature only when its value changes.

    Designed for staircase-like features such as the smoothed heart rate,
    which is held constant between R-peaks. Filtering duplicates yields one
    history point per beat, so the AR model's index axis equals the beat axis
    and _estimate_dt() naturally recovers the RR interval.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_value: Optional[float] = None
        self._beat_count = 0

    def add_feature(self, feature_value: Optional[float],
                    timestamp: Optional[float] = None) -> None:
        if feature_value is None:
            return                      # HR deriver warm-up: skip
        if self._last_value is not None and feature_value == self._last_value:
            return                      # same beat, duplicate sample: skip
        self._last_value = feature_value
        self._beat_count += 1
        super().add_feature(feature_value, timestamp)

    def beat_count(self) -> int:
        """Number of distinct beats absorbed so far (diagnostic)."""
        return self._beat_count


class PassthroughFeatureDeriver(StreamingFeatureDeriver):
    """Identity deriver: feature = raw sample. Lets the AR model
    forecast the ECG waveform itself (used in RAW mode only)."""

    def __init__(self) -> None:
        self._sample: Optional[float] = None

    def add_sample(self, sample: float, timestamp: float = None) -> None:
        self._sample = sample

    def get_feature(self) -> Optional[float]:
        return self._sample

    def reset(self) -> None:
        self._sample = None


class NoopChangePointDetector(StreamingChangePointDetector):
    """Never-firing detector stub for RAW mode (the HR anomaly detector
    must not be fed raw ECG samples)."""

    def update(self, feature_value: Optional[float],
               raw_sample: Optional[float] = None) -> bool:
        return False


def log_forecast(predictor: BeatGatedARPredictor,
                 lead_beats: tuple = (1, 5, 10, 20)) -> None:
    """
    Log the current HR forecast at selected lead times (in beats).

    Call periodically or after the stream ends; a lightweight stand-in for the
    planned forecast-vs-actual evaluation loop.
    """
    forecast = predictor.get_forecast()
    if forecast is None:
        logger.info("No forecast yet (warm-up), beats seen: %d",
                    predictor.beat_count())
        return

    logger.info(
        "HR forecast @ beat %d (dt=%.3f s/step):",
        predictor.beat_count(), forecast["dt"],
    )
    idx = [min(b, len(forecast["mean"])) - 1 for b in lead_beats]
    for b, i in zip(lead_beats, idx):
        logger.info(
            "  +%2d beats (t=%.1fs): %.1f bpm  [%.1f, %.1f]",
            b, forecast["time"][i], forecast["mean"][i],
            forecast["lower"][i], forecast["upper"][i],
        )


# ---------------------------------------------------------------------------
# HR MODE: forecast heart rate, one model update per beat
# ---------------------------------------------------------------------------

def run_hr_forecast_example() -> None:
    """
    Run the standard ECG R-peak pipeline with a beat-gated AR forecaster
    of heart rate. Anomaly period: t=20-25 s exercises on_change_point().
    """
    logger.info("Starting ECG heart-rate forecast demo")

    # 1. Visualization setup
    fig, ax = plt.subplots(figsize=(12, 5))
    setup = PlottingSetup(
        fig=fig,
        ax=ax,
        title="ECG R-Peak Detection with AR Heart-Rate Forecast",
        ylim=(-2, 2),
    )

    # 2. Data source with a temporary HR anomaly (change point at ~20 s)
    data_source = create_ecg_with_temporary_anomaly(start=20.0, stop=25.0)

    # 3. Standard HR feature deriver (Pan-Tompkins style)
    hr_deriver = HRFeatureDeriver(
        fs=360.0,
        min_rr_sec=0.2,
        threshold_factor=0.7,
    )

    # 4. Standard anomaly detector
    detector = HeartRateAnomalyDetector(
        threshold_std=3.0,
        warmup_beats=15,
        confirmation_count=3,
        recovery_count=5,
        baseline_adaptation=0.0,
    )

    # 5. NEW: beat-gated AR predictor
    #    order/window/horizon are in BEATS thanks to the gating.
    #    refit_every=1: refit on every beat (tiny lstsq, negligible cost).
    predictor = BeatGatedARPredictor(
        order=8,           # short memory: HR varies smoothly beat-to-beat
        window_size=60,    # fit on the last ~60 beats
        horizon=20,        # forecast 20 beats (~20 RR intervals) ahead
        refit_every=1,
        ci_z=1.96,         # ~95% band
    )

    # 6. Renderer (vline: ignores forecast, shows ECG + anomaly markers)
    renderer = HeartRateForecastRenderer(
        line_color='black',
        vline_color='red',
        vline_style='--',
        vline_width=2.0,
        signal_label='ECG',
        vline_label='Anomaly',
        hr_color='steelblue',
        forecast_color='darkorange',
        title_template='Heart Rate: {feature:.1f} bpm',
    )
    # 7. Simulator with predictor slot
    sim = StreamingSimulator(
        plotting_setup=setup,
        feature_deriver=hr_deriver,
        change_point_detector=detector,
        renderer=renderer,
        data_source=data_source,
        predictor=predictor,      # <-- the new slot
        window_duration_sec=5.0,
        max_history=10000,
        interval_ms=30,
    )

    logger.info("Starting stream (close the plot window to finish)")
    sim.start()

    # 8. Post-mortem: report what the model learned
    logger.info("=" * 60)
    logger.info("Stream finished. %d distinct beats absorbed.",
                predictor.beat_count())
    log_forecast(predictor)
    logger.info("Full trace with per-sample forecasts is in %s",
                logging_setup.log_file)


# ---------------------------------------------------------------------------
# RAW MODE: forecast the ECG waveform itself
# ---------------------------------------------------------------------------

def run_raw_forecast_example() -> None:
    """
    Run an AR forecaster directly on the raw ECG waveform.

    NOTE: draw the forecast you additionally need a forecast-aware renderer
    (e.g. an extended ModelLineRenderer). With VerticalLineRenderer this mode
    validates the pipeline and logs forecasts but does not draw them.
    """
    logger.info("Starting raw-ECG waveform forecast demo")

    fig, ax = plt.subplots(figsize=(12, 5))
    setup = PlottingSetup(
        fig=fig,
        ax=ax,
        title="Raw ECG with AR Waveform Forecast",
        ylim=(-2, 2),
    )

    data_source = create_ecg_data_source()

    deriver = PassthroughFeatureDeriver()
    detector = NoopChangePointDetector()

    # At 360 Hz one cardiac cycle spans ~300-400 samples. The AR order must
    # cover at least one full cycle, or the forecast collapses to the mean
    # instead of tracing the PQRST shape forward.
    predictor = StreamingARPredictor(
        order=300,          # ~1 cardiac cycle
        window_size=3000,   # ~10 cycles of fit history
        horizon=200,        # ~0.55 s of waveform lookahead
        refit_every=200,    # refit ~5x/sec; 2500x301 lstsq is tens of ms
        ci_z=1.96,
    )

    renderer = VerticalLineRenderer(
        line_color='black',
        signal_label='ECG',
        title_template='Raw EGC — AR(300) forecast: {feature:.3f}',
    )

    sim = StreamingSimulator(
        plotting_setup=setup,
        feature_deriver=deriver,
        change_point_detector=detector,
        renderer=renderer,
        data_source=data_source,
        predictor=predictor,
        window_duration_sec=5.0,
        max_history=10000,
        interval_ms=30,
    )

    sim.start()

    logger.info("Raw-ECG forecast demo finished")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pick a mode:
    run_hr_forecast_example()
    # run_raw_forecast_example()
