"""Abstract base classes for the streaming framework."""

__author__ = "F.Feenstra"

from abc import ABC, abstractmethod
from typing import Optional, Any, List
import numpy as np


class StreamingFeatureDeriver(ABC):
    """
    Abstract base class for real-time feature extraction from streaming data.
    
    This class defines the interface for components that process incoming data
    samples sequentially to derive meaningful features (e.g., statistical metrics,
    signal characteristics, or event detections). Implementations maintain internal
    state to compute features incrementally without needing to store the entire
    history of raw data.
    
    Methods:
        add_sample(sample, timestamp): Feed a new data point into the processor.
        get_feature(): Retrieve the most recently calculated feature value.
        reset(): Clear all internal state and buffers.
    
    """
    @abstractmethod
    def add_sample(self, sample: float, timestamp: float = None) -> None:
        """
        Add a new sample to the processor.
        
        Processes the incoming data point and updates internal state.
        This method is called frequently in the streaming loop.
        
        Args:
            sample (float): The new data value to process.
            timestamp (float, optional): The timestamp associated with the sample.
                                         Used for time-based calculations if supported.
        """
        pass
    
    @abstractmethod
    def get_feature(self) -> Optional[float]:
        """
        Get the latest derived feature value.
        
        Returns the most recently computed feature based on the samples processed
        so far. Returns None if insufficient data exists to compute a feature.
        
        Returns:
            Optional[float]: The calculated feature value, or None if unavailable.
        """
        pass
    
    def reset(self) -> None:
        """
        Reset internal state.
        
        Clears all buffers, counters, and accumulated statistics.
        Called when starting a new data stream or segment.
        """
        pass


class StreamingChangePointDetector(ABC):
    """
    Abstract base class for detecting distribution shifts (change points) in streaming data.
    
    This class defines the interface for algorithms that monitor a sequence of feature
    values and identify moments where the underlying statistical properties of the
    data change significantly (e.g., mean shift, variance change, concept drift).
    
 
    Attributes:
        drift_detected (bool): Property indicating if a change point was recently detected.
                               Defaults to False; implementations override this logic.
    
    Methods:
        update(feature_value): Process a new feature and return detection result.
        drift_detected: Read-only property for current drift status.
    
    """
    
    @abstractmethod
    def update(self, feature_value: Optional[float], raw_sample: Optional[float] = None) -> bool:
        """
        Update detector with new feature value and optional raw sample.
        
        Args:
            feature_value: The latest feature value.
            raw_sample: The raw signal value (optional, for residual detection).
        
        Returns:
            bool: True if a change point is detected.
        """
        pass
    
    @property
    def drift_detected(self) -> bool:
        """
        Check if drift/change was detected.
        
        Returns the current status of the detector. This property reflects the
        outcome of the most recent `update()` call or the cumulative state
        depending on the implementation strategy.
        
        Returns:
            bool: True if a change point is currently flagged, False otherwise.
        """
        return False


class StreamingRenderer(ABC):
    """
    Abstract base class for real-time visualization of streaming data.
    
    This class defines the interface for components responsible for rendering
    dynamic plots in a streaming environment. It separates the logic of data
    preparation from the rendering engine, allowing for efficient updates
    without redrawing the entire figure.
    
    Methods:
        initialize(ax): Set up the initial plot structure on the provided axes.
        update(...): Refresh plot data for the current time window.
        cleanup(): Release resources and clear references.
    
    """
    
    @abstractmethod
    def initialize(self, ax: Any) -> List[Any]:
        """
        Create initial plot elements.
        
        Sets up the necessary matplotlib artists (lines, scatter plots, etc.)
        on the provided axes. This is called once when the visualization starts.
        
        Args:
            ax (Any): The matplotlib Axes instance to draw on.
        
        Returns:
            List[Any]: A list of artist objects (e.g., Line2D, PathCollection)
                       that should be tracked for efficient updates (blitting).
        """
        pass
    
    @abstractmethod
    def update(self, times: np.ndarray, samples: np.ndarray, 
               features: np.ndarray, change_points: np.ndarray,
               window_duration_sec: float, 
               forecast: Optional[dict[str, np.ndarray]] = None) -> List[Any]:
        """
        Update plot elements with new data.
        
        Refreshes the data for the existing artists to reflect the latest
        streaming data within the specified time window.
        
        Args:
            times (np.ndarray): Array of timestamps for the samples.
            samples (np.ndarray): Array of raw data values.
            features (np.ndarray): Array of derived feature values (may contain None).
            change_points (np.ndarray): Array of timestamps where changes were detected.
            window_duration_sec (float): Duration of the visible time window in seconds.
            ------ prediction part ------
            forecast: (dict) a datamatrix with the predictions for instance
            {
            "time":   np.array([10.05, 10.06, 10.07, ..., 10.54]),  # future timestamps
            "mean":   np.array([0.41,  0.43,  0.46,  ..., 0.85 ]),  # point forecast per step
            "lower":  np.array([0.35,  0.36,  0.37,  ..., 0.52 ]),  # lower bound per step
            "upper":  np.array([0.47,  0.50,  0.55,  ..., 1.18 ]),  # upper bound per step
            "dt":     0.01,                                          # est. seconds per step
            }
        
        Returns:
            List[Any]: Updated list of artist objects that have been modified.
                       Used by animation frameworks to determine what to redraw.

        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources if needed.
        
        Clears references to plot elements and performs any necessary teardown
        to prevent memory leaks, especially important in long-running streaming
        applications.
        """
        pass


# 16 november add streaming predictor
class StreamingPredictor(ABC):
    """
    Abstract base class for forecasting future feature values from a stream.

    Fed sequentially with derived features in the processing loop; produces a
    multi-step forecast of where the feature is heading.

    The models learns from recent history, 
    so after a sudden change in your data it keeps predicting the old behavior 
    until stale samples slowly age out — a window-stability-vs-adaptability 
    trade-off formalized in the concept drift literature 
    (Gama et al., ACM Computing Surveys 2014). 
    Adaptive shrinking of the history window after a detected change is the core idea 
    of ADWIN (Bifet & Gavaldà, 2007). Our on_change_point() is a pragmatic version 
    of that: when the detector says "something changed," 
    the predictor drops the older half of its history and refits, 
    adapting in one step instead of over a whole window. 
    Truncating rather than wiping everything keeps enough data to forecast immediately, 
    avoiding a blind spot right when accuracy matters most.
    """

    @abstractmethod
    def add_feature(self, feature_value: Optional[float],
                    timestamp: Optional[float] = None) -> None:
        """Feed the latest derived feature into the predictor."""

    @abstractmethod
    def get_forecast(self) -> Optional[dict[str, np.ndarray]]:
        """Return the current forecast, or None during warm-up."""

    def on_change_point(self) -> None:
        """Hook: called when the change point detector fires. Default: no-op."""

    def reset(self) -> None:
        """Clear all internal state."""