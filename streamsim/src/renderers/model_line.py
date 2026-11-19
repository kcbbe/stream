from os import times
from typing import List, Any, Optional
import numpy as np
from streamsim.src.core.interfaces import StreamingRenderer 

class ModelLineRenderer(StreamingRenderer):
    def __init__(
        self,
        line_color: str = 'blue',       # Raw signal color
        line_width: float = 2,
        line_alpha: float = 0.6,
        model_color: str = 'green',      # Modeled signal color
        model_width: float = 1,
        model_alpha: float = 0.8,
        model_style: str = '--',       # Dashed line for model
        vline_color: str = 'purple',   # Vertical line color for change points
        vline_alpha: float = 0.5,
        vline_width: float = 1.5,
        vline_label: str = 'Anomaly',
        signal_label: str = 'Raw Signal',
        model_label: str = 'Modeled Signal',
        show_legend: bool = True,
        title_template: str = "Raw vs Model — Latest: {feature:.3f}"
    ):
        # Store styles
        self.line_color = line_color
        self.line_width = line_width
        self.line_alpha = line_alpha
        self.model_color = model_color
        self.model_width = model_width
        self.model_alpha = model_alpha
        self.model_style = model_style
        self.vline_color = vline_color
        self.vline_alpha = vline_alpha
        self.vline_width = vline_width
        self.vline_label = vline_label
        self.signal_label = signal_label
        self.model_label = model_label
        self.show_legend = show_legend
        self.title_template = title_template
        
        self.line = None          # Raw signal
        self.model_line = None    # Model
        self.title_artist = None
        self.ax = None
        self.artists = []

    def initialize(self, ax: Any) -> List[Any]:
        self.ax = ax

        # 1. Raw Signal Line
        self.line, = ax.plot([], [], lw=self.line_width,
                            color=self.line_color, label=self.signal_label,
                            alpha=self.line_alpha)

        # 2. Model Line
        self.model_line, = ax.plot([], [], lw=self.model_width,
                                color=self.model_color, linestyle=self.model_style,
                                alpha=self.model_alpha, label=self.model_label)

        # 3. Vertical Lines for anomalies (single artist, NaN-separated segments)
        self.vlines, = ax.plot([], [], lw=self.vline_width,
                            color=self.vline_color, alpha=self.vline_alpha,
                            label=self.vline_label)

        self.title_artist = ax.set_title("")

        # added for prediction
        self.forecast_line, = ax.plot([], [], lw=1.5, color='orange',
                              linestyle='-.', alpha=0.9, label='Forecast')
        self.band = ax.fill_between([], [], [],
                            color='orange', alpha=0.15, label='_nolegend_')

        if self.show_legend:
            ax.legend()

        self.artists = [self.line, self.model_line, self.vlines, self.forecast_line, self.band, self.title_artist]
        return self.artists

    def update(self, times, samples, features, change_points, window_duration_sec, forecast):
        if len(times) == 0:
            return self.artists

        # Filter data for the time window
        mask = times >= times[-1] - window_duration_sec
        window_times = times[mask]
        window_samples = samples[mask]
        window_features = features[mask] if len(features) == len(samples) else np.array([])
        

        # Update Raw Signal
        self.line.set_data(window_times, window_samples)
        
        # Update Smoothed Model
        # Ensure we have data for the smoothed line
        if len(window_features) > 0:
            self.model_line.set_data(window_times, window_features)
            self.model_line.set_visible(True)
        else:
            self.model_line.set_visible(False)

        if len(change_points) > 0:
            visible_mask = change_points >= times[-1] - window_duration_sec
            visible_cps = change_points[visible_mask]

            if len(visible_cps) > 0:
                ymin, ymax = self.ax.get_ylim()
                x_data = []
                y_data = []
                for cp in visible_cps:
                    # Each vertical line is two points: bottom to top
                    # NaN separates segments so they don't connect horizontally
                    x_data.extend([cp, cp, np.nan])
                    y_data.extend([ymin, ymax, np.nan])

                self.vlines.set_data(x_data, y_data)
                self.vlines.set_visible(True)
            else:
                self.vlines.set_data([], [])
                self.vlines.set_visible(False)
        else:
            self.vlines.set_data([], [])
            self.vlines.set_visible(False)

        # added for prediction
        if forecast is not None and len(forecast.get("mean", [])) > 0:
            self.forecast_line.set_data(forecast["time"], forecast["mean"])
            self.band.remove()
            self.band = self.ax.fill_between(
                forecast["time"], forecast["lower"], forecast["upper"],
                color='orange', alpha=0.15)
            self.forecast_line.set_visible(True)
        else:
            self.forecast_line.set_visible(False)

        # Update Title
        if len(features) > 0 and self.title_artist is not None:
            latest_feature = features[-1]
            self.title_artist.set_text(self.title_template.format(feature=latest_feature))
        
        return self.artists

    def cleanup(self):
        self.line = None
        self.model_line = None
        self.vlines = None       
        self.title_artist = None
        self.ax = None
        self.artists = []