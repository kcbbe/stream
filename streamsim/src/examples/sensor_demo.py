"""
Sensor Anomaly Detection Demonstration Script.

a_data.csv (raw data)
    ↓
csv_to_arrays (records, category_names)
    ↓
create_uni_signal() → (raw_sample, timestamp)
    ↓
StreamingSimulator._processing_loop()
    ├─ sample = raw_sample                    ← RAW SIGNAL
    └─ feature_deriver.add_sample(raw_sample)
         ↓
    SmoothedSignalDeriver (calculates moving average)
         ↓
    feature = deriver.get_feature()           ← MODEL VALUE
    ↓
    change_point_detector.update(feature, raw_sample=sample)
         ↓
    ResidualThresholdDetector (calculates |raw - model|)
         ↓
    is_change = (residual > threshold)        ← ANOMALY FLAG
    ↓
Queue.put({"sample": sample, "feature": feature, "change_points": [...]})
    ↓
StreamingSimulator._update_plot()
    ├─ samples[]  = all raw samples
    ├─ features[] = all model values
    └─ change_points[] = anomaly timestamps
    ↓
ModelLineRenderer.update()
    ├─ line.set_data(times, samples)          ← Draws RAW line (black)
    ├─ model_line.set_data(times, features)   ← Draws MODEL line (red dashed)
    └─ vlines.set_data(...)                   ← Draws VERTICAL lines (purple)

    __author__ = "F.Feenstra, J. Beenen"
"""

# -------------------------------------------------------------------
# set up logging before importing other modules
# -------------------------------------------------------------------

import logging
from streamsim.src.core.config import LoggingSetup, PlottingSetup
from streamsim.src.core.config import LOG_DIR

logging_setup = LoggingSetup(
    level=logging.INFO,
    log_file=f"{LOG_DIR}/sensor_demo.log",
    timestamp_filename=True,             # Set to True if you want unique filenames per run
    console=True
)
logging_setup.setup_logging()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# import main components for this demo
# -------------------------------------------------------------------

import yaml
# Data Source: Reads from a CSV file with 'timestamp', 'category', and 'value' columns
from streamsim.src.sources.a_data_source import csv_to_arrays, create_uni_signal, get_dataframe
# deriver = SmoothedSignalDeriver (calculates moving average)
from streamsim.src.features.simple import SmoothedSignalDeriver
import matplotlib.pyplot as plt
# renderer = ModelLineRenderer (plots raw signal + smoothed model)
from streamsim.src.renderers.model_line import ModelLineRenderer
# detector = AbsoluteThresholdDetector (simple absolute threshold exceeder detection)
    # OPTION 1: AbsoluteThresholdDetector
from streamsim.src.detectors.absolute_threshold import AbsoluteThresholdDetector
    # OPTION 2: PlateauDetector
from streamsim.src.detectors.plateau import PlateauDetector
# ty together in a simulator
from streamsim.src.core.simulator import StreamingSimulator

import numpy as np

# Load configuration from YAML file
with open("./config.yaml", "r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)


def sensor_demo():

    # 1. Setup Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    setup = PlottingSetup(
        fig=fig, 
        ax=ax, 
        title="Model line", 
        ylim=(int(config["y_min"]), int(config["y_max"]))
    )

    if config["y_dynamic"]:
        def update_ylim(_event):
            """Fit the y-axis to the values in the currently visible window. Written by Copilot, with minor adjustments by Jennefer Beenen."""
            x_min, x_max = ax.get_xlim()
            values = []
            first_time = True
            for line in ax.lines:
                if str(line.get_label()).lower() != 'anomaly':
                    x_data = np.asarray(line.get_xdata())
                    y_data = np.asarray(line.get_ydata())

                    if first_time:
                        first_time = False

                    if len(x_data) != len(y_data):
                        continue
                    visible = (x_data >= x_min) & (x_data <= x_max)
                    values.extend(y_data[visible])

            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return

            lower, upper = values.min(), values.max()
            padding = max((upper - lower) * 0.05, 1e-9)
            ax.set_ylim(lower - padding, upper + padding)

        fig.canvas.mpl_connect("draw_event", update_ylim)

    # 2. Initialize Data Source
    records, category_names = csv_to_arrays(
            file_path= config["sensor_file"],
            timestamp= config["timestamp_column"],
            category= config["category_column"],
            value= config["value_column"],
    )
    source = create_uni_signal(
        selected_parameter= config["selected_parameter"],
        records= records,
        category_names= category_names,
    )

    # 3. Configure Signal Smoother
    deriver = SmoothedSignalDeriver(window_size = 100)
    
    # 4. Configure Anomaly Detector (get threshold dataframe and filter on univariate selected_parameter)
    all_thresholds = get_dataframe(config["threshold_file"])
    parameter_thresholds = all_thresholds[all_thresholds[config["category_column"]] == config["selected_parameter"]]

    # OPTION 1: AbsoluteThresholdDetector
    # detector = AbsoluteThresholdDetector(
    #     sorted((
    #         parameter_thresholds["threshold_start"].values[0],
    #         parameter_thresholds["threshold_stop"].values[0]
    #     ))
    # )

    # OPTION 2: PlateauDetector
    detector = PlateauDetector(history=25)

    # 5. Configure Renderer
    renderer = ModelLineRenderer(
        show_legend=True,
        signal_label='Raw Signal',
        model_label='Smoothed Model',
        vline_color= 'red',
        title_template="Latest Value: {feature:.1f}"
    )   

    # 6. Initialize and Start Simulator
    sim = StreamingSimulator(
        plotting_setup=setup,
        feature_deriver=deriver,
        change_point_detector=detector,
        renderer=renderer,
        data_source=source,
        window_duration_sec=50,
        max_history=10000,
        interval_ms=1
    )

    logger.info("Starting Model Demo...")
    sim.start()

if __name__ == "__main__":
    sensor_demo()
