"""Module that contains all code for calibration feature detection."""

from __future__ import annotations
from PyCBD.pipelines import CBDPipeline
from PyGeiger.detector import GeigerDetector
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
import logging


class FeatureDetector:
    """Object used for detecting a specified calibration feature in an image.

    Which feature the detector tries to detect is determined upon initialization. Afterwards it can be used
    to detect the same calibration feature in several images. If neither board_size nor marker are given, only the
    relative object space location of the checkerboard corners can be determined.
    """

    def __init__(self, space_between_features: float, board_size: Optional[Tuple[int, int]] = None,
                 marker: Optional[Tuple[int, int]] = None, **kwargs) -> None:
        """Class constructor.

        :param space_between_features: Checker size in mm.
        :param board_size: Size of the board in (rows, columns)
        :param marker: Position of the marker if there is a marker present.
        :param kwargs: Additional arguments for :py:class:`CBDPipeline` (expand, predict, out_of_image)
        """
        self._logger = logging.getLogger(__name__)
        self.space_between_features = space_between_features
        self.board_size = board_size
        self.marker = marker
        self.units = 'mm'
        if marker is not None:
            print("Markers have not been implemented yet.")
        self.detector = CBDPipeline(GeigerDetector(), **kwargs)

    def detect_feature(self, image: npt.NDArray) -> CalibrationFeature:
        """Attempt to detect the specified feature in the image.

        :param image: Grayscale or BGR color image.
        :raises RuntimeError: When there is something wrong with the image.
        :returns: Object that contains all feature data for the image.
        """

        if self.board_size is None:
            score, board_uv, board_xy = self.detector.detect_checkerboard(image)
        else:
            score, board_uv, board_xy = self.detector.detect_checkerboard(image, self.board_size)
        object_points = np.concatenate((board_xy, np.zeros((board_xy.shape[0], 1))), -1)
        calibration_feature = CalibrationFeature(score, board_uv.astype(np.float32),
                                                 (object_points*self.space_between_features).astype(np.float32))

        return calibration_feature


class CalibrationFeature:
    """Object that stores all the data related to the calibration feature detection of an image."""

    def __init__(self, score: int, image_points: npt.NDArray, object_points: npt.NDArray) -> None:
        """Class constructor."""

        self.score: int = score
        """Whether feature detection was successful or not."""

        self.image_points: npt.NDArray = image_points
        """Array containing all image points. Is None if detection was not successful."""

        self.object_points: npt.NDArray = object_points
        """Array containing all object points. Is None if detection was not successful."""
