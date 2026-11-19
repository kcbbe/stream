"""
Matplotlib Streaming Renderer with Marker Point Visualization

This module provides a streaming renderer class for visualizing time-series
data with automatic annotation of detected Marker points. It is designed
for integration with real-time data pipelines where Marker points need to
be highlighted dynamically as new data arrives.

The renderer maintains a sliding time window, automatically filtering data
to display only the most recent samples within the configured duration.
Marker points are interpolated to match the signal line's y-values
at the detected timestamps.

Features:
    - Dynamic title updates with feature values
    - Configurable styling for signal lines and marker points
    - Efficient incremental updates (no full redraws)
    - Automatic visibility management for marker points
    - Resource cleanup via explicit cleanup() method

Important Dependencies:
    - streamsim.src.core.interfaces.StreamingRenderer: Base interface

Author: F.Feenstra

Example:
    >>> import matplotlib.pyplot as plt
    >>> from streamsim.src.renderers.matplotlib_line import MatplotlibLineRenderer
    >>> renderer = MatplotlibLineRenderer(
    ...     line_color='blue',
    ...     marker_style='ro',
    ...     marker_alpha=0.8,
    ...     show_legend=True,
    ...     marker_label='Peak',
    ...     title_template="Sinus Wave Peaks — Latest Value: {feature:.1f}"
    ... )
    >>> fig, ax = plt.subplots()
    >>> artists = renderer.initialize(ax)
    >>> # In streaming loop:
    >>> # updated_artists = renderer.update(times, samples, features, change_points, window_duration)
"""

from typing import List, Any, Optional
import numpy as np
from streamsim.src.core.interfaces import StreamingRenderer 

class MatplotlibLineRenderer(StreamingRenderer):
    """
    Default matplotlib line chart renderer for streaming data visualization.
    
    This class implements the StreamingRenderer interface to provide real-time
    visualization of streaming signals with configurable styling and marker point
    annotation. It supports dynamic title updates based on feature values and
    flexible legend configuration for both signal and marker elements.
    
    Designed for use with the streaming framework's data pipeline, this renderer
    efficiently updates plot elements without redrawing the entire figure,
    making it suitable for high-frequency data streams.
    """
    
    def __init__(
        self,
        line_color: str = 'blue',
        line_width: float = 2,
        marker_style: str = 'ro',
        marker_alpha: float = 0.7,
        signal_label: str = 'Signal',
        marker_label: str = 'Change',
        show_legend: bool = True,
        title_template: str = "Streaming Data — Feature: {feature:.4f}"
    ):
        self.line_color = line_color
        self.line_width = line_width
        self.marker_style = marker_style
        self.marker_alpha = marker_alpha
        self.signal_label = signal_label
        self.marker_label = marker_label
        self.show_legend = show_legend
        self.title_template = title_template
        
        self.line = None
        self.marker = None
        self.title_artist = None
        self.ax = None
        self.artists = []
    
    def initialize(self, ax: Any) -> List[Any]:
        """
        Create initial plot elements on the provided axes.
        
        Sets up the signal line, markers, and dynamic title on the
        given matplotlib axes. This method should be called once before the
        streaming loop begins.
        
        Args:
            ax (Any): The matplotlib Axes instance to draw on. Typically obtained
                      from plt.subplots() or fig.add_subplot().
        
        Returns:
            List[Any]: List of artist objects [line, marker, title_artist] that
                       should be tracked for efficient updates (e.g., blitting).
        
        Example:
        >>> renderer = MatplotlibLineRenderer(
        ...     line_color='blue',
        ...     marker_style='ro',
        ...     marker_alpha=0.8,
        ...     show_legend=True,
        ...     marker_label='Peak',
        ...     title_template="Sinus Wave Peaks — Latest Value: {feature:.1f}"
        ... )
        >>> fig, ax = plt.subplots()
        >>> artists = renderer.initialize(ax)
        """
        self.ax = ax
        self.line, = ax.plot([], [], lw=self.line_width, 
                             color=self.line_color, label=self.signal_label)
        
        self.marker, = ax.plot([], [], self.marker_style, 
                               alpha=self.marker_alpha, label=self.marker_label)
        
        self.title_artist = ax.set_title("")
        
        if self.show_legend:
            ax.legend()
        
        self.artists = [self.line, self.marker, self.title_artist]
        return self.artists
    
    def update(
        self,
        times: np.ndarray,
        samples: np.ndarray,
        features: np.ndarray,
        change_points: np.ndarray,
        window_duration_sec: float,
        forecast: Optional[dict[str, np.ndarray]] = None
    ) -> List[Any]:
        """
        Update plot elements with new streaming data.
        
        Args:
            times (np.ndarray): Array of timestamps.
            samples (np.ndarray): Array of signal values.
            features (np.ndarray): Array of feature values.
            change_points (np.ndarray): Array of change point timestamps.
            window_duration_sec (float): Visible time window duration.
        
        Returns:
            List[Any]: Updated artist objects.
        
        Example:
            >>> updated_artists = renderer.update(times, samples, features, change_points, window_duration
        """
        if len(times) == 0:
            return self.artists
        
        # Update signal line
        mask = times >= times[-1] - window_duration_sec
        self.line.set_data(times[mask], samples[mask])
        
        # Update change point markers
        if len(change_points) > 0:
            visible_mask = change_points >= times[-1] - window_duration_sec
            visible_cps = change_points[visible_mask]
            
            if len(visible_cps) > 0:
                y_values = np.interp(visible_cps, times, samples)
                self.marker.set_xdata(visible_cps)
                self.marker.set_ydata(y_values)
                self.marker.set_visible(True)
            else:
                self.marker.set_visible(False)
        else:
            self.marker.set_visible(False)
        
        # Update title with latest feature value
        if len(features) > 0 and self.title_artist is not None:
            latest_feature = features[-1]
            self.title_artist.set_text(self.title_template.format(feature=latest_feature))
        
        return self.artists
    
    def cleanup(self) -> None:
        """
        Release resources and clear references.
        This method can be called when the simulation is stopped to clean up any resources and prevent memory leaks.
        """
        self.line = None
        self.marker = None
        self.title_artist = None
        self.ax = None
        self.artists = []