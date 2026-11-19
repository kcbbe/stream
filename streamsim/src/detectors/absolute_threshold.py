"""
Absolute Threshold Detector

Detects when the raw signal falls outside the configured threshold range.
"""

from typing import Optional, Tuple
from streamsim.src.core.interfaces import StreamingChangePointDetector
from streamsim.src.core.logger import get_logger  # Import logger

# Get a logger specific to this module
logger = get_logger(__name__)


class AbsoluteThresholdDetector(StreamingChangePointDetector):
    """
    Flags a detection when (raw_sample > threshold_stop) or (raw_sample < threshold_start).
    """
    
    def __init__(self, threshold: Tuple[float, float]):
        self.threshold = threshold # TODO: What if there is just one threshold?
        self._sample_count = 0
        self._drift_detected = False
        
        logger.info(f"[AbsoluteThresholdDetector] Initialized: thresholds={self.threshold}")
    
    def update(self, feature_value: Optional[float], raw_sample: Optional[float] = None) -> bool:
        self._sample_count += 1
        self._drift_detected = False
                
        # 2. Validate Inputs
        if feature_value is None or raw_sample is None:
            logger.warning(f"[AbsoluteThresholdDetector] Missing data: raw={raw_sample}, model={feature_value}")
            return False
                   
        # 4. Log Every Sample (DEBUG level) or just Anomalies (INFO level)
        # Uncomment the line below if you want to see EVERY calculation (can be verbose!)
        # logger.debug(f"[AbsoluteThresholdDetector] Check: | raw={raw_sample:.4f} | model={feature_value:.4f}")
        
        # 5. Check Threshold
        # if raw_sample outside threshold (lower, upper):
        if (raw_sample < self.threshold[0]) or (raw_sample > self.threshold[1]):
            self._drift_detected = True
            
            # Log the ANOMALY clearly
            logger.info(
                f"!! ANOMALY DETECTED: raw_sample={raw_sample:.4f} > threshold={self.threshold} | "
                f"Count={self._sample_count}"
            )
            return True
        
        return False
    
    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
        
    def reset(self) -> None:
        logger.info("[AbsoluteThresholdDetector] Resetting state.")
        self._sample_count = 0
        self._drift_detected = False
