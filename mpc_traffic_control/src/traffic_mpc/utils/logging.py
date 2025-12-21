"""
Logging and Telemetry Utility.
Configures hierarchical logging and structured data recording.
"""
import logging
import logging.config
import os
from datetime import datetime
from traffic_mpc.config.settings import LoggingConfig

def setup_logging(cfg: LoggingConfig):
    """Configures the Python logging system."""
    os.makedirs(cfg.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.log_dir, f"mpc_run_{timestamp}.log")

    config_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_file,
                'level': cfg.level,
                'formatter': 'standard',
                'encoding': 'utf8'
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': cfg.level,
        }
    }
    logging.config.dictConfig(config_dict)