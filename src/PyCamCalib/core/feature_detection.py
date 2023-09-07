"""Module that contains all code for calibration feature detection."""

from __future__ import annotations
from PyCBD.pipelines import CBDPipeline
from PyCamCalib.core.exceptions import ImageError
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
import logging


class FeatureDetector:
    """Object used for detecting a specified calibration feature in an image.

    Which feature the detector tries to detect is determined upon initialization. Afterwards it can be used
    to detect the same calibration feature in several images. If neither board_size nor marker are given, only the
    relative object space location of the checkerboard corners can be determined.

    :ivar space_between_features: Real world checker size in mm.
    :ivar board_size: Size of the board in (rows, columns)
    :ivar marker: Position of the marker if there is a marker present.
    :ivar units: Real world units.
    :ivar detector: The actual checkerboard detector used for detection.

    """

    def __init__(self, space_between_features: float, board_size: Optional[Tuple[int, int]] = None,
                 marker: Optional[Tuple[int, int]] = None, **kwargs) -> None:
        """Class constructor.

        :param space_between_features: Checker size in mm.
        :param board_size: Size of the board in (rows, columns)
        :param marker: Position of the marker if there is a marker present.
        :param kwargs: Additional arguments for the PyCBD checkerboard detection pipeline
           (expand, predict, out_of_image)
        """
        self._logger = logging.getLogger(__name__)
        self.space_between_features = space_between_features
        self.board_size = board_size
        self.marker = marker
        self.units = 'mm'
        if marker is not None:
            raise NotImplementedError("Markers have not been implemented yet.")
        self.detector = CBDPipeline(**kwargs)

    def detect_feature(self, image: npt.NDArray) -> CalibrationFeature:
        """Attempt to detect the specified feature in the image.

        :param image: Grayscale or BGR color image.
        :raises ImageError: When there was an issue with the calibration image.
        :returns: Object that contains all feature data for the image.
        """

        try:
            if self.board_size is None:
                score, board_uv, board_xy = self.detector.detect_checkerboard(image)
            else:
                score, board_uv, board_xy = self.detector.detect_checkerboard(image, self.board_size)
        except TypeError as e:
            raise ImageError("There was an issue with the calibration image: " + str(e))
        except ValueError as e:
            raise ImageError("There was an issue with the calibration image: " + str(e))
        if score != 0:
            object_points = np.concatenate((board_xy, np.zeros((board_xy.shape[0], 1))), -1)
            calibration_feature = CalibrationFeature(score, board_uv.astype(np.float32),
                                                     (object_points*self.space_between_features).astype(np.float32))
        else:
            calibration_feature = CalibrationFeature(score, np.array([]), np.array([]))

        return calibration_feature


class CalibrationFeature:
    """Object that stores all the data related to the calibration feature detection of an image.

    :ivar score: Whether feature detection was successful or not 0 means the detection failed, 1
       means the xy coordinates are relative, 2 means the xy coordinates are absolute
    :ivar image_points: Array containing all image points. Is empty if detection was not successful.
    :ivar object_points: Array containing all object points. Is empty if detection was not successful.
    """

    def __init__(self, score: int, image_points: npt.NDArray, object_points: npt.NDArray) -> None:
        """Class constructor."""
        self.score: int = score
        self.image_points: npt.NDArray = image_points
        self.object_points: npt.NDArray = object_points
