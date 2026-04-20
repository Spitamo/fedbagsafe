import logging
import sys


def setup_logging(level: str = 'INFO', verbose: bool = True, 
                 log_file: str = None):
    """setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # siilence PL
    pytorch_logger = logging.getLogger("pytorch_lightning")
    pytorch_logger.setLevel(logging.WARNING)

    return logging.getLogger(__name__)