import logging
import sys


def setup_logging(level: str = 'INFO', verbose: bool = True,
                  log_file: str = None):
    """setup logging config"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    log_level = getattr(logging, level.upper()) if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # silence lightning logs
    pytorch_logger = logging.getLogger("pytorch_lightning")
    pytorch_logger.setLevel(logging.WARNING)

    return logging.getLogger(__name__)