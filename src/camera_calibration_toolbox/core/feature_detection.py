"""Module that contains all code for calibration feature detection."""

from __future__ import annotations
from camera_calibration_toolbox.core.exceptions import TagError, ImageError
import cv2
import numpy as np
import numpy.typing as npt
import logging


class FeatureDetector:
    """Object used for detecting a specified calibration feature in an image.

    Which feature the detector tries to detect is determined by the tag upon initialization. Afterwards it can be used
    to detect the same calibration feature in several images.
    """

    def __init__(self, tag: str) -> None:
        """Class constructor.

        :param tag: Tag that describes the feature which needs to be detected.
        :raises TagError: When an invalid tag is used. This can occur when the wrong prefix is used, the tag is not
           long enough or the tail doesn't contain numeric data.
        :raises TypeError: When tag is not a string.
        """
        self._logger = logging.getLogger(__name__)
        if isinstance(tag, str):
            if tag.startswith('NCH'):
                try:
                    self.board_size = (int(tag[7:9]), int(tag[9:11]))
                    """Amount of inner corners in (rows, columns)."""

                    self.space_between_features = float(tag[11:])
                    """Distance between feature points in :py:data:`units`."""
                except ValueError:
                    raise TagError("Invalid tag.")
                except IndexError:
                    raise TagError("Invalid tag.")
                else:
                    self.feature_type = 'checker'
                    """Which type of feature the detector will look for."""

                    self.units = 'mm'
                    """Units in object-space between feature points."""
            else:
                raise TagError("Invalid tag.")
        else:
            raise TypeError("``tag`` needs to be a string.")

    def detect_feature(self, image: npt.NDArray, normalize: bool = False,
                       invert: bool = False) -> CalibrationFeature:
        """Attempt to detect the specified feature in the image.

        :param image: One- or multi-channel image.
        :param normalize: Whether to normalize the image or not, defaults to False.
        :param invert: Whether to invert the image or not, defaults to False.
        :raises ImageError: When image is not 2- or 3-dimensional.
        :raises TypeError: When the type of image is not a numpy array.
        :returns: Object that contains all feature data for the image.
        """
        if isinstance(image, np.ndarray):
            dim = image.shape
            n_dim = len(dim)
            if n_dim == 2 or len(dim) == 3:
                if len(dim) == 3:
                    if dim[2] == 3:
                        self._logger.info("BGR image converted to grayscale")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        self._logger.info("First frame of image hypercube selected")
                        image = image[:, :, 0]

                if normalize:
                    image = self.normalize_image(image)
                else:
                    data_type = image.dtype
                    if data_type != 'uint8':
                        image = self.recast_to_uint8(image)

                if invert:
                    image = np.invert(image)

                calibration_feature = self.detect_checkerboard(image)
            else:
                raise ImageError("``image`` should have 2 or 3 dimensions, the given image has  " + str(n_dim)
                                 + " dimensions.")
        else:
            raise TypeError("The type of ``image`` should be a numpy array")

        return calibration_feature

    def normalize_image(self, image: npt.NDArray) -> npt.NDArray[np.uint8]:
        """Normalize the image."""
        image = image.astype('double')
        minimum = np.amin(image)
        maximum = np.amax(image)
        return ((image - minimum) / (maximum - minimum) * 255).astype('uint8')

    def recast_to_uint8(self, image: npt.NDArray) -> npt.NDArray[np.uint8]:
        """Recast the image to uint8."""
        if image.dtype == 'uint16':
            image = image.astype('double')
            image = (image / 65535 * 255).astype('uint8')
            self._logger.info("Converted image to uint8 without normalization.")
        else:
            image = image.astype('double')
            if np.amax(image) <= 1 and np.amin(image) >= 0:
                image = (image * 255).astype('uint8')
                self._logger.info("Converted image to uint8 without normalization.")
            else:
                image = self.normalize_image(image)
                self._logger.info("Converted image to uint8 with normalization.")
        return image

    def detect_checkerboard(self, image: npt.NDArray[np.uint8]) -> CalibrationFeature:
        """Detect regular checkerboard in an image."""
        score, image_points = cv2.findChessboardCorners(image, self.board_size, None)
        if score:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
            image_points = np.squeeze(cv2.cornerSubPix(image, image_points, (11, 11), (-1, -1), criteria))
            rows = self.board_size[0]
            columns = self.board_size[1]
            object_points = np.zeros((rows * columns, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
            object_points = object_points * self.space_between_features
            feature_image = cv2.drawChessboardCorners(np.stack((image,) * 3, axis=-1), self.board_size, image_points,
                                                      score)
            calibration_feature = CalibrationFeature(score, feature_image, image_points, object_points)
        else:
            feature_image = np.stack((image,) * 3, axis=-1)
            calibration_feature = CalibrationFeature(score, feature_image, None, None)

        return calibration_feature


class CalibrationFeature:
    """Object that stores all the data related to the calibration feature detection of an image."""

    def __init__(self, score: bool, feature_image: npt.NDArray[np.uint8], image_points: npt.NDArray[np.float32] | None,
                 object_points: npt.NDArray[np.float32] | None) -> None:
        """Class constructor."""

        self.score = score
        """Whether feature detection was successful or not."""

        self.image_points = image_points
        """Array containing all image points. Is None if detection was not successful."""

        self.object_points = object_points
        """Array containing all object points. Is None if detection was not successful."""

        self.feature_image = feature_image
        """(converted) RGB image that was used for feature detection with feature points if detection was successful."""
