"""Module for very basic logging configuration of the package."""

import logging


def configure_logger(level: str = 'WARNING') -> None:
    """Performs a very basic logging configuration for the package logger, which will output to sys.stderr.

    The level can be set to 'CRITICAL', 'ERROR', 'WARNING' or 'INFO', which correspond to the levels in the
    python logging module.

    :param level: Which logging level should be used.
    :raises TypeError: If the type of level is not a string.
    """
    if isinstance(level, str):
        logger = logging.getLogger('camera_calibration_toolbox')
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.name = "basic_handler"
            formatter = logging.Formatter('%(thread)d - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if level == 'CRITICAL':
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        elif level == 'ERROR':
            logger.setLevel(logging.ERROR)
            handler.setLevel(logging.ERROR)
        elif level == 'INFO':
            logger.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)
        elif level == 'DEBUG':
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
    else:
        raise TypeError("`level` should be a string.")
