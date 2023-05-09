"""Module that contains custom exceptions used by the camera calibration toolbox."""


class ImageError(Exception):
    """Raised when there is an issue with the image."""

    pass


class CalibrationError(Exception):
    """Raised when there is an issue with calibration."""

    pass