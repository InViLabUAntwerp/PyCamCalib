import cv2
import numpy as np
from Calibration.CalibrationFeatureDetector import CalibrationFeatureDetector
from Calibration.CalibrationParameters import CalibrationParameters


class CameraCalibrator:

    def __init__(self):

        self.calibrator = ''
        self.image_points_array = []
        self.object_points_array = []
        self.feature_images = []
        self.image_indices = []
        self.per_view_err = []
        self.height = 0
        self.width = 0

    def calibrate_camera(self, image_array, tag, calibrator='opencv', invert=False, normalize=False):

        if not isinstance(tag, str):
            raise ValueError('"tag" should be a string.')
        if not isinstance(calibrator, str):
            raise ValueError('"calibrator" should be a string.')
        if calibrator != 'opencv':
            raise ValueError("Provided calibrator: '" + calibrator + "' has not been implemented.")

        self.construct_feature_arrays(image_array, tag, invert, normalize)
        if len(self.image_points_array) == 0:
            raise RuntimeError('Failed to detect features in all images, unable to perform calibration.')
        elif len(self.image_points_array) < 11:
            print('Warning: only detected features for ' + str(len(self.image_points_array)) +
                  ' images. Features from at least 11 images are necessary for an accurate calibration.')

        self.calibrator = calibrator
        if calibrator == 'opencv':
            calibration_parameters = self.opencv_calibration()

        return calibration_parameters

    def construct_feature_arrays(self, image_array, tag, invert, normalize):

        self.image_points_array = []
        self.object_points_array = []
        self.feature_images = []
        self.image_indices = []
        feature_detector = CalibrationFeatureDetector(tag)
        if len(image_array.shape) == 3:
            self.height, self.width, n_images = image_array.shape
            for idx in range(n_images):
                score, image_points, object_points, feature_image \
                    = feature_detector.detect_features(image_array[:, :, idx], invert, normalize)
                self.feature_images.append(feature_image)
                if score:
                    self.image_points_array.append(image_points)
                    self.object_points_array.append(object_points)
                    self.image_indices.append(idx)
                else:
                    print('Failed feature detection on image nr' + str(idx+1) + '.')
        elif len(image_array.shape) == 4:
            self.height, self.width, n_channels, n_images = image_array.shape
            for idx in range(n_images):
                score, image_points, object_points, feature_image \
                    = feature_detector.detect_features(image_array[:, :, :, idx], invert, normalize)
                self.feature_images.append(feature_image)
                if score:
                    self.image_points_array.append(image_points)
                    self.object_points_array.append(object_points)
                    self.image_indices.append(idx)
                else:
                    print('Failed feature detection on image nr' + str(idx + 1) + '.')
        else:
            raise ValueError('Incompatible image array. Only image arrays with len(image_array).shape = 3 or 4 can '
                             'be used. The provided image array has ' + str(len(image_array.shape)) + ' dimensions.')

    def opencv_calibration(self):

        rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs, t_vecs, intrinsics_std, extrinsics_std, per_view_err \
            = cv2.calibrateCameraExtended(self.object_points_array, self.image_points_array, (self.width, self.height),
                                          None, None)

        self.per_view_err = np.squeeze(per_view_err)
        dist_coeffs = np.squeeze(dist_coeffs)
        intrinsics_std = np.squeeze(intrinsics_std)
        extrinsics_std = np.squeeze(extrinsics_std)

        calibration_parameters = CalibrationParameters()
        calibration_parameters.set_parameters_opencv(rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs, t_vecs,
                                                     intrinsics_std, extrinsics_std, self.width, self.height)

        return calibration_parameters

    def ransac_calibration(self):

        pass

    def repeat_calibration_remove_samples(self, indices):

        if len(self.image_points_array) - len(indices) <= 0:
            raise RuntimeError('Length of "indices" to be removed is the same as the amount of samples.')
        elif len(self.image_points_array) - len(indices) < 11:
            print('Warning: only ' + str(len(self.image_points_array)) +
                  ' samples will be left. At least 11 samples are necessary for an accurate calibration.')
        self.image_points_array = [i for j, i in enumerate(self.image_points_array) if j not in indices]
        self.object_points_array = [i for j, i in enumerate(self.object_points_array) if j not in indices]
        self.image_indices = [i for j, i in enumerate(self.image_indices) if j not in indices]

        if self.calibrator == 'opencv':
            calibration_parameters = self.opencv_calibration()

        return calibration_parameters