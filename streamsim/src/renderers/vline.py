"""
Matplotlib Streaming Renderer with Change Point Visualization 
by usage of vertical lines.

This module provides a streaming renderer class for visualizing time-series
data with automatic annotation of detected change points (anomalies). It is
designed for integration with real-time data pipelines where change points
need to be highlighted dynamically as new data arrives.

The renderer maintains a sliding time window, automatically removing outdated
visual elements and adding new ones for change points within the visible range.
This ensures efficient memory usage during long-running streaming sessions.

Features:
    - Dynamic title updates with feature values
    - Automatic cleanup of out-of-window change point markers
    - Configurable styling for signal lines and change point indicators

Important Dependencies:
    - streamsim.src.core.interfaces.StreamingRenderer: Base interface

Author: F.Feenstra

Example:
    >>> import matplotlib.pyplot as plt
    >>> from streamsim.src.renderers.vline import VerticalLineRenderer
    >>> renderer = VerticalLineRenderer(
    ...     line_color='black',
    ...     vline_color='red',
    ...     title_template='Heart Rate: {feature:.1f} bpm'
    ... )
    >>> fig, ax = plt.subplots()
    >>> artists = renderer.initialize(ax)
    >>> # In streaming loop:
    >>> # updated_artists = renderer.update(times, samples, features, change_points, window_duration, forecast)
"""


from typing import List, Any, Optional
import numpy as np
from streamsim.src.core.interfaces import StreamingRenderer


class VerticalLineRenderer(StreamingRenderer):
    """
    Streaming renderer that displays vertical lines at detected change points.
    
    This class extends the StreamingRenderer interface to provide real-time
    visualization of time-series signals with annotated change points. Each
    change point is marked with a configurable vertical line that persists
    only while within the visible time window.
    
    The renderer tracks displayed change points internally to prevent duplicate
    annotations when the same change point is received across multiple update
    cycles. Out-of-window change points are automatically removed to maintain
    rendering performance.
    """
    
    def __init__(
        self,
        #signal styling
        line_color: str = 'blue',
        line_width: float = 2,
        signal_label: str = 'Signal',
        #vertical line styling
        vline_color: str = 'red',
        vline_style: str = '--',
        vline_width: float = 1.5,
        vline_alpha: float = 0.7,
        vline_label: str = 'Anomaly',
        
        show_legend: bool = True,
        title_template: str = "Streaming Data — Feature: {feature:.4f}"
    ):
        self.line_color = line_color
        self.line_width = line_width
        self.signal_label = signal_label
        
        self.vline_color = vline_color
        self.vline_style = vline_style
        self.vline_width = vline_width
        self.vline_alpha = vline_alpha
        self.vline_label = vline_label
        
        self.show_legend = show_legend
        self.title_template = title_template
        
        self.line = None
        self.vlines = []  # List of vertical line artists
        self.title_artist = None
        self.ax = None
        self.artists = []
        
        # Track displayed change points to avoid duplicates
        self._displayed_changes = set()
        
    
    def initialize(self, ax: Any) -> List[Any]:
        """
        Create initial plot elements on the provided axes.
        
        Args:
            ax (Any): The matplotlib Axes instance to draw on.
        
        Returns:
            List[Any]: List of artist objects for animation tracking.

        Example:
            >>> fig, ax = plt.subplots()
            >>> renderer = VerticalLineRenderer()
            >>> artists = renderer.initialize(ax)
        """
        self.ax = ax
        self.line, = ax.plot([], [], lw=self.line_width, 
                             color=self.line_color, label=self.signal_label)
        
        # Create a proxy artist for legend (actual vlines created dynamically)
        self._legend_proxy, = ax.plot([], [], 
                                       color=self.vline_color,
                                       linestyle=self.vline_style,
                                       lw=self.vline_width,
                                       label=self.vline_label)
        
        self.title_artist = ax.set_title("")
        
        if self.show_legend:
            ax.legend()
        
        self.artists = [self.line, self.title_artist]
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
        
        # Get current y-limits for vertical lines
        y_min, y_max = self.ax.get_ylim()
        
        # Determine visible window
        window_start = times[-1] - window_duration_sec
        window_end = times[-1]
        
        # Remove old vertical lines outside the window
        for vline in self.vlines[:]:
            x_pos = vline.get_xdata()[0]
            if x_pos < window_start:
                vline.remove()
                self.vlines.remove(vline)
                self._displayed_changes.discard(x_pos)
        
        # Add new vertical lines for change points in window
        if len(change_points) > 0:
            visible_cps = change_points[change_points >= window_start]
            
            for cp in visible_cps:
                # Round to avoid floating point duplicate issues
                cp_key = round(cp, 4)
                
                if cp_key not in self._displayed_changes:
                    # Create vertical line spanning full y-range
                    vline = self.ax.axvline(
                        x=cp,
                        color=self.vline_color,
                        linestyle=self.vline_style,
                        linewidth=self.vline_width,
                        alpha=self.vline_alpha
                    )
                    self.vlines.append(vline)
                    self._displayed_changes.add(cp_key)
        
        # Update title with latest feature value
        if len(features) > 0 and self.title_artist is not None:
            latest_feature = features[-1]
            if latest_feature is not None:
                self.title_artist.set_text(
                    self.title_template.format(feature=latest_feature)
                )
        
        return self.artists + self.vlines
    

    def cleanup(self) -> None:
        """
        Release resources and clear references.
        
        Removes all vertical line artists from the axes, clears internal
        tracking sets, and nullifies references to prevent memory leaks
        during long-running streaming sessions.
        
        Note:
            Call this method when the renderer is no longer needed to ensure
            proper garbage collection of matplotlib artist objects.
        """
        for vline in self.vlines:
            vline.remove()
        self.vlines.clear()
        self._displayed_changes.clear()
        
        self.line = None
        self.title_artist = None
        self.ax = None
        self.artists = []