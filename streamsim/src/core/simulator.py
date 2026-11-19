"""
Multi-Threaded Streaming Simulator

This module provides a general-purpose framework for real-time visualization and
analysis of streaming time-series data, featuring a producer-consumer architecture
with pluggable components for feature extraction and change point detection.

Architecture:
    The simulator employs a dual-thread design: a background worker thread
    continuously pulls data from a source, derives features, and runs change
    point detection, pushing results to a thread-safe queue. The main thread
    consumes queued updates and renders via an animated visualization, ensuring
    UI responsiveness independent of processing load.

Use Cases:
    - Real-time sensor monitoring with live anomaly visualization
    - Educational demonstrations of signal processing pipelines

Important Dependencies:
    - threading: Background processing thread
    - queue.Queue: Thread-safe communication between producer and consumer
    - collections.deque: Efficient circular buffers for history management
    - matplotlib.animation.FuncAnimation: Animated rendering engine
    - streamsim.src.core.interfaces: Pluggable component interfaces
    - streamsim.src.core.config.PlottingSetup: Visualization configuration

    
Example:    
    >>> sim = StreamingSimulator(
    ...    plotting_setup=setup,
    ...    feature_deriver=deriver,
    ...    change_point_detector=detector,
    ...    renderer=renderer,
    ...    data_source=sinus_source,
    ...    window_duration_sec=5.0,  
    ...    max_history=5000,         
    ...    interval_ms=100           
    )
    >>> sim.start()
    # [Matplotlib window opens with live streaming visualization]


Author: F.Feenstra

"""
import numpy as np
import threading
from typing import Optional, Any, List
import time
from queue import Queue, Empty
from typing import Callable, Optional, Tuple, List
from collections import deque
from streamsim.src.core.interfaces import StreamingFeatureDeriver, StreamingChangePointDetector, StreamingRenderer
from streamsim.src.core.config import PlottingSetup
from streamsim.src.core.interfaces import StreamingPredictor

import logging
logger = logging.getLogger(__name__) 

class StreamingSimulator:
    """
    General framework for streaming time-series visualization.
    
    Provides multi-threaded architecture with pluggable components for
    feature extraction and change point detection.
    """
    
    def __init__(
        self,
        plotting_setup: PlottingSetup,
        feature_deriver: StreamingFeatureDeriver,
        change_point_detector: StreamingChangePointDetector,
        renderer: StreamingRenderer = None,
        data_source: Callable[[], Optional[Tuple[float, float]]] = None,
        window_duration_sec: float = 30.0,
        interval_ms: int = 50,
        max_history: int = 2000,
        # added predictor
        predictor: StreamingPredictor = None
    ):
        self.plotting_setup = plotting_setup
        self.feature_deriver = feature_deriver
        self.change_point_detector = change_point_detector
        self.data_source = data_source
        self.window_duration_sec = window_duration_sec
        self.interval_ms = interval_ms
        self.max_history = max_history
        # added predictor
        self.predictor = predictor
        
        # Import here to avoid circular dependency
        from streamsim.src.renderers.matplotlib_line import MatplotlibLineRenderer
        self.renderer = renderer or MatplotlibLineRenderer()

        # Buffers (MAIN THREAD ONLY)
        self.sample_buffer = deque(maxlen=max_history)
        self.time_buffer = deque(maxlen=max_history)
        self.feature_buffer = deque(maxlen=max_history)
        self.change_point_times = deque(maxlen=max_history)

        # Queue (thread communication)
        self.queue = Queue(maxsize=10000)

        # State
        self.is_running = False
        self.animation = None
        self.processing_thread = None

    def _processing_loop(self) -> None:
        """
        Background worker thread: orchestrates the data ingestion and analysis pipeline.

        This method runs continuously in a separate thread while the simulator is active.
        It acts as the producer in a producer-consumer architecture, responsible for:
        
        1. **Data Acquisition**: Fetching raw samples from the configured data source.
        2. **Feature Extraction**: Passing samples to the `StreamingFeatureDeriver` to
           compute relevant features.
        3. **Change Point Detection**: Updating the `StreamingChangePointDetector` with
           derived features to identify anomalies or shifts.
        4. **State Management**: Tracking consecutive outliers to confirm change points
           and managing the internal `change_timestamps` list.
        5. **Queue Dispatch**: Packaging the sample, timestamp, feature, and detection
           status into a dictionary and pushing it to the thread-safe queue for the
           main rendering thread to consume.

        The loop terminates when `is_running` is set to False or when the
        data source returns None, signaling the end of the stream.

        """
    
        self.change_timestamps = []
        
        while self.is_running:
            sample_data = self.data_source()
            if sample_data is None:
                break

            timestamp = None
            try:
                sample, timestamp = sample_data
                self.feature_deriver.add_sample(sample, timestamp)
            except (ValueError, TypeError):
                sample = sample_data
                self.feature_deriver.add_sample(sample)

            feature = self.feature_deriver.get_feature()
            
            # added for streamingpredictor
            forecast = None
            if self.predictor is not None and feature is not None:
                self.predictor.add_feature(feature, timestamp)
                forecast = self.predictor.get_forecast()
            ## end for streamingpredictor

            is_change = self.change_point_detector.update(feature_value=feature, raw_sample=sample)
            
            logger.debug(f"[Simulator] Feature={feature}, is_change={is_change}")
            
            if is_change:
                if not hasattr(self, '_anomaly_line_added') or not self._anomaly_line_added:
                    self.change_timestamps.append(timestamp)
                    self._anomaly_line_added = True
                    if self.predictor is not None:
                        self.predictor.on_change_point() 
                    logger.debug(f"[Simulator] ADDED change point: t={timestamp:.2f}s")

            else:
                self._anomaly_line_added = False

            if not self.queue.full():
                self.queue.put({
                    "sample": sample,
                    "timestamp": timestamp,
                    "feature": feature,
                    "is_change": is_change,
                    "change_points": np.array(self.change_timestamps),
                    # added for predictor
                    "forecast": forecast
                })

            time.sleep(0.001)


    def _init_plot(self) -> List[Any]:
        """Initialize plot elements for animation."""
        artists = self.renderer.initialize(self.plotting_setup.ax)
        self.plotting_setup.configure()
        return artists
    

    def _update_plot(self, frame: int) -> List[Any]:
        """
        Main thread: consumes queued data and renders the updated visualization.

        This method is invoked by `FuncAnimation` on each animation frame and serves
        as the consumer in the producer-consumer architecture. It performs:

        1. **Queue Consumption**: Drains up to 200 pending messages from the thread-safe
           queue, aggregating data into internal circular buffers.
        2. **Buffer Management**: Appends incoming samples, timestamps, features, and
           change point markers to their respective deques.
        3. **Data Preparation**: Converts buffered data into NumPy arrays suitable for
           efficient rendering.
        4. **Visualization Update**: Delegates to the configured `StreamingRenderer`
           to refresh the plot with the latest data and detected change points.
        5. **Viewport Adjustment**: Updates the x-axis limits to maintain a sliding
           window view centered on the most recent data.

        Args:
            frame (int): Frame index provided by FuncAnimation (unused but required
                by the animation callback signature).

        Returns:
            List[Any]: A list of matplotlib artists modified during the update,
                enabling efficient blitting if enabled.

        Note:
            This method must remain lightweight to preserve smooth animation.
            All computationally intensive work is offloaded to `_processing_loop`.
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # Track change points from queue
        change_points_list = []
        # added for predictor
        latest_forecast = None
        
        count = 0
        while count < 200:
            try:
                msg = self.queue.get_nowait()
            except Empty:
                break
            
            if msg.get("end"):
                self.is_running = False
                break

            if msg.get("forecast") is not None:
                latest_forecast = msg["forecast"]

            # added for prediction
            self.sample_buffer.append(msg["sample"])
            self.time_buffer.append(msg["timestamp"])
            
            if msg["feature"] is not None:
                self.feature_buffer.append(msg["feature"])
            
            # Collect change points from queue message
            if "change_points" in msg and len(msg["change_points"]) > 0:
                change_points_list = list(msg["change_points"])
            
            count += 1

        if not self.time_buffer:
            return []
        
        times = np.fromiter(self.time_buffer, float)
        samples = np.fromiter(self.sample_buffer, float)
        # features = np.fromiter(self.feature_buffer, float) if self.feature_buffer else np.array([])
        features = np.array(self.feature_buffer, dtype=object) if self.feature_buffer else np.array([])
        cps = np.array(change_points_list) if change_points_list else np.array([])

        # added forecast
        artists = self.renderer.update(times, samples, features, cps, self.window_duration_sec, latest_forecast)

        latest = times[-1]

        right_edge = latest
        if latest_forecast is not None and len(latest_forecast.get("time", [])) > 0:
            right_edge = max(right_edge, float(latest_forecast["time"][-1]))

        if latest <= self.window_duration_sec:
            self.plotting_setup.ax.set_xlim(0, self.window_duration_sec)
        else:
            self.plotting_setup.ax.set_xlim(latest - self.window_duration_sec, right_edge)
        #latest = times[-1]
        #if latest <= self.window_duration_sec:
        #    self.plotting_setup.ax.set_xlim(0, self.window_duration_sec)
        #else:
        #    self.plotting_setup.ax.set_xlim(latest - self.window_duration_sec, latest)
        
        return artists


    def start(self) -> None:
        """
        Launches the streaming simulation and enters the main event loop.

        This method initializes and starts both the background processing thread
        and the matplotlib animation engine, coordinating the full simulation
        pipeline from data ingestion to real-time visualization.

        Initialization Steps:
            1. Sets the `is_running` flag to True, enabling the worker thread.
            2. Spawns a daemon thread executing `_processing_loop` for continuous
               data acquisition, feature derivation, and change point detection.
            3. Configures the `FuncAnimation` engine with the plotting setup,
               binding `_update_plot` as the frame callback and `_init_plot`
               for initial render state.
            4. Enters the matplotlib main loop via `plt.show()`, blocking until
               the window is closed or `stop()` is called.

        Threading Model:
            - Worker thread: Handles all data processing and queue production
            - Main thread: Manages GUI rendering and queue consumption
            - Both threads communicate via a thread-safe `Queue` instance

        Lifecycle:
            The simulation runs indefinitely until:
            - The user closes the plot window
            - `stop()` is called programmatically
            - The data source signals completion (returns None)

        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.is_running = True

        self.plotting_setup.fig.canvas.mpl_connect('close_event', self._on_close)

        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()

        self.animation = FuncAnimation(
            self.plotting_setup.fig,
            self._update_plot,
            init_func=self._init_plot,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False
        )

        plt.show()


    def _on_close(self, event) -> None:
        """Handle figure close event."""
        logger.debug("[Simulator] Window closed, stopping...")
        self.stop()


    def stop(self) -> None:
        """
        Terminates the streaming simulation and releases resources.

        This method initiates a clean shutdown sequence for both the background
        processing thread and the matplotlib animation engine, ensuring all
        resources are properly released and no orphaned threads remain.

        Shutdown Sequence:
            1. Sets `is_running` to False, signaling the worker thread to exit
               its processing loop on the next iteration.
            2. Waits for the processing thread to join with a 1-second timeout,
               allowing it to finish any in-flight operations.
            3. Stops the `FuncAnimation` event source to halt frame callbacks.
            4. Invokes the renderer's cleanup routine to release any allocated
               graphics resources or figure handles.

        Thread Safety:
            This method is thread-safe and can be called from any thread. It
            uses the shared `is_running` flag to coordinate shutdown with the
            background worker, avoiding race conditions or deadlocks.

        Post-Execution State:
            After `stop()` completes:
            - The processing thread is terminated (or timed out)
            - The animation engine is halted
            - Internal buffers retain their last state (call `reset()` if needed)
            - The plot window remains open until manually closed

        Note:
            Always call `stop()` before closing the application to prevent
            resource leaks or hanging threads. For a fresh start, call `reset()`
            on components before invoking `start()` again.
        """
        logger.debug("Simulation is stopped")
        self.is_running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=1)

        if self.animation:
            self.animation.event_source.stop()

        self.renderer.cleanup()