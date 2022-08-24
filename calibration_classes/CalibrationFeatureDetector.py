import cv2
import numpy as np
from checkerboard import detect_checkerboard


class CalibrationFeatureDetector:

    def __init__(self, tag):

        # eg: string = "MCH0604091418.2"
        # 1 - 3: MCH: Checkerboard with Markers
        # 4 - 7: 0604: Marker top circle at position 6, 4
        # 8 - 11: 0914: boardsize = (9, 14) (rows, columns)
        # 12 - end: 18.5: checker size( in mm)
        #
        # eg: string = "NCH0000050630"
        # 1 - 3: NCH: Normal Checkerboard
        # 4 - 7: unused
        # 8 - 11: 0506: boardsize = (5, 6) (rows, columns)
        # 12 - end: 30: checker size( in mm)
        try:
            if tag[0:3] == 'NCH':
                self.type = 'checker'
                self.units = 'mm'
                self.board_size = (int(tag[7:9]), int(tag[9:11]))
                self.space_between_features = float(tag[11:])
            else:
                raise LookupError('Invalid tag')
        except:
            raise LookupError('Invalid tag')

    def detect_features(self, image, invert=False, normalize=False):

        dimensions = image.shape
        if len(dimensions) == 3:
            if dimensions[2] == 3:
                print('BGR image converted to grayscale')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                print('First frame of image hypercube selected')
                image = image[:, :, 0]

        if normalize:
            image = self.normalize_image(image)
        else:
            data_type = image.dtype
            if data_type != 'uint8':
                image = self.recast_to_uint8(image)

        if invert:
            image = np.invert(image)

        if self.type == 'checker':
            return self.detect_checkerboard(image)

    def normalize_image(self, image):

        image = image.astype('double')
        minimum = np.amin(image)
        maximum = np.amax(image)
        return ((image - minimum) / (maximum - minimum) * 255).astype('uint8')

    def recast_to_uint8(self, image):

        if image.dtype == 'uint16':
            image = image.astype('double')
            image = (image / 65535 * 255).astype('uint8')
            print('Converted image to uint8 without normalization.')
        else:
            image = image.astype('double')
            if np.amax(image) <= 1 and np.amin(image) >= 0:
                image = (image * 255).astype('uint8')
                print('Converted image to uint8 without normalization.')
            else:
                image = self.normalize_image(image)
                print('Converted image to uint8 with normalization.')
        return image

    def detect_checkerboard(self, image):

        h, w = image.shape
        h_s = int(h / 4)
        w_s = int(w / 4)
        resized = cv2.resize(image, (w_s, h_s))
        score, corners = cv2.findChessboardCorners(resized, self.board_size, None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        corners = np.squeeze(cv2.cornerSubPix(image, corners * 4, (51, 51), (-1, -1), criteria))
        corners_image = cv2.drawChessboardCorners(np.stack((image,)*3, axis=-1), self.board_size, corners, score)
        rows = self.board_size[0]
        columns = self.board_size[1]
        object_points = np.zeros((rows * columns, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        object_points = object_points * self.space_between_features

        return score, corners, object_points, corners_image

