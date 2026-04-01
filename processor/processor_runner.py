"""
Processor runner: orchestrates feature extraction → anomaly detection.
Runs every 30 minutes, offset 30 minutes after AIS ingestor.
"""

from processor.ais_anomaly_detector import detect_ais_anomalies
from processor.ais_feature_extractor import run_ais_feature_extraction
from processor.anomaly_detector import run_anomaly_detection
from processor.feature_extractor import run_feature_extraction
from processor.monitoring_window import run_monitoring_window_check
from processor.satellite_feature_extractor import run_satellite_feature_extraction
from processor.satellite_anomaly_detector import detect_satellite_anomalies
from processor.rail_feature_extractor import run_rail_feature_extraction
from processor.rail_anomaly_detector import detect_rail_anomalies
from shared.db import init_db
from shared.logger import get_logger

logger = get_logger(__name__)


def run() -> dict:
    # Price + news features and anomalies
    features = run_feature_extraction()
    anomalies = run_anomaly_detection()

    # AIS features and anomalies
    ais_features = run_ais_feature_extraction()
    ais_anomalies = detect_ais_anomalies()

    # Satellite + aircraft features and anomalies
    sat_features = run_satellite_feature_extraction()
    sat_anomalies = detect_satellite_anomalies()

    # Rail corridor features + anomaly detection
    rail_features = run_rail_feature_extraction()
    rail_anomalies = detect_rail_anomalies()

    # Monitoring window checkpoints
    monitoring = run_monitoring_window_check()

    summary = {
        "features": features,
        "anomalies": anomalies,
        "ais_features": ais_features,
        "ais_anomaly_count": len(ais_anomalies),
        "satellite_features": sat_features,
        "satellite_anomaly_count": len(sat_anomalies),
        "rail_features": rail_features,
        "rail_anomaly_count": len(rail_anomalies),
        "monitoring": monitoring,
    }
    logger.info("processor_run_complete", **summary)
    return summary


if __name__ == "__main__":
    init_db()
    result = run()
    print(result)
