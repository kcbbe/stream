"""
Plateau Detector

Detects a plateau in the raw signal.

Detection Method:
    The detector maintains a rolling history of recent samples and computes the
    mean, standard deviation (SD), and slope. A new sample is flagged as a plateau if it
    the slope is (or smaller than) `near_zero` and the SD is (or smaller than) `near_zero`.~~

Important Dependencies:
    - collections.deque: Efficient circular buffer
    - streamsim.src.core.interfaces.StreamingChangePointDetector: Base interface

Author: F. Feenstra & J. Beenen
# TODO: How to handle differences in y-signal span/spread? (current method is more sensitive to signal with small spread in total signal.)
"""

from collections import deque
import numpy as np
from streamsim.src.core.interfaces import StreamingChangePointDetector
from streamsim.src.core.logger import get_logger  # Import logger

# Get a logger specific to this module
logger = get_logger(__name__)


class PlateauDetector(StreamingChangePointDetector):
    """
    Flags a detection when (plateau = (the slope is (or smaller than) `near_zero`) and (the SD is (or smaller than) `near_zero`).)~~
)
    """
    
    def __init__(
            self, 
            near_zero: float = 0.0001,
            history: int = 50
        ):
        """
        Initialize the PlateauDetector with a configurable sensitivity threshold (`near_zero`).
        
        Args:
            near_zero (float): The threshold for considering the slope and standard deviation as "near zero".
                               Default: 0.0001.
            history (int): The number of recent samples to maintain in the rolling history for analysis.
                           Default: 50.
        """
        self.near_zero = near_zero
        self.history_len = int(history)
        self.history = deque(maxlen=self.history_len)
        self._sample_count = 0
        self._drift_detected = False

        logger.info(f"[PlateauDetector] Initialized: near_zero= {self.near_zero}; history= {self.history}")

    def update(self, feature_value: float, raw_sample=None) -> bool:
        """
        Process a new sample and determine if it represents a plateau.
        A plateau = (the absolute value of the slope is (or smaller than) `near_zero`) and (the absolute value of the standard deviation is (or smaller than) `near_zero`).)

        Args:
            feature_value (float): The model-derived feature value (not used in this detector).
            raw_sample (float): The new sample value to evaluate.

        Returns:
            bool: True if the sample is flagged as a plateau, False otherwise.
        
        """
        self._sample_count += 1
        self._drift_detected = False
        self.history.append(raw_sample)

        # Detector logics
        # Check if history is initiated
        if len(self.history) > self.history_len -1:
            arr = np.array(self.history)
            # Exclude current sample from statistics (use all but last)
            mean = arr[:-1].mean()
            std = arr[:-1].std()
            slope = np.polyfit(
                np.arange(arr[:-1].shape[0]),
                arr[:-1],
                1
            )[0]

            # 4. Log Every Sample (DEBUG level) or just Anomalies (INFO level)
            # Uncomment the line below if you want to see EVERY calculation (can be verbose!)
            logger.debug(f"[PlateauDetector] Check: raw_sample= {raw_sample:.4f}; mean= {mean:.4f}; std= {std:.4f}; slope= {slope:.4f}; count= {self._sample_count}")
            
            # 5. Check Threshold
            # if standard deviation and slope in `history` is smaller than provided threshold (=`near_zero`):
            if (abs(std) <= self.near_zero) & (abs(slope) <= self.near_zero): # `slope == 0` does not work due to floatpoint error.

                # Log the ANOMALY clearly
                logger.info(
                    f"!! ANOMALY DETECTED: raw_sample= {raw_sample:.4f}; mean= {mean:.4f}; std= {std:.4f}; slope= {slope:.4f} | "
                    f"Count={self._sample_count}"
                )
                return True
                
        return False



    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
        
    def reset(self) -> None:
        logger.info("[PlateauDetector] Resetting state.")
        self._sample_count = 0
        self._drift_detected = False
