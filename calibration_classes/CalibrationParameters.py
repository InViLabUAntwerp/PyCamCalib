import numpy as np
import csv


class CalibrationParameters:
    def __init__(self):
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.s = 0
        self.fx_std = 0
        self.fy_std = 0
        self.cx_std = 0
        self.cy_std = 0
        self.s_std = 0
        self.rms_reproj_error = 0
        self.radial_dist_coeffs = np.zeros(3)
        self.tangential_dist_coeffs = np.zeros(2)
        self.radial_dist_coeffs_std = np.zeros(3)
        self.tangential_dist_coeffs_std = np.zeros(2)
        self.width = 0
        self.height = 0

    def get_afov(self):

        fov_x = np.rad2deg(2 * np.arctan2(self.width, 2 * self.fx))
        fov_y = np.rad2deg(2 * np.arctan2(self.height, 2 * self.fy))

        return fov_x, fov_y

    def get_intrinsics_matrix_opencv(self):

        intrinsics_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        return intrinsics_matrix

    def get_distortion_coeffs_opencv(self):

        return np.array([self.radial_dist_coeffs[0], self.radial_dist_coeffs[1], self.tangential_dist_coeffs[0],
                         self.tangential_dist_coeffs[1], self.radial_dist_coeffs[2]])

    def set_parameters_opencv(self, rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs, t_vecs, intrinsics_std,
                              extrinsics_std, width, height):

        self.fx = intrinsics_matrix[0, 0]
        self.fy = intrinsics_matrix[1, 1]
        self.cx = intrinsics_matrix[0, 2]
        self.cy = intrinsics_matrix[1, 2]
        self.s = 0
        self.fx_std = intrinsics_std[0]
        self.fy_std = intrinsics_std[1]
        self.cx_std = intrinsics_std[2]
        self.cy_std = intrinsics_std[3]
        self.s_std = 0
        self.rms_reproj_error = rms_reproj_error
        self.radial_dist_coeffs[0] = dist_coeffs[0]
        self.radial_dist_coeffs[1] = dist_coeffs[1]
        self.radial_dist_coeffs[2] = dist_coeffs[4]
        self.tangential_dist_coeffs[0] = dist_coeffs[2]
        self.tangential_dist_coeffs[0] = dist_coeffs[3]
        self.radial_dist_coeffs_std[0] = intrinsics_std[4]
        self.radial_dist_coeffs_std[0] = intrinsics_std[5]
        self.radial_dist_coeffs_std[0] = intrinsics_std[8]
        self.tangential_dist_coeffs_std[0] = intrinsics_std[6]
        self.tangential_dist_coeffs_std[0] = intrinsics_std[7]
        self.width = width
        self.height = height

    def load_parameters_csv(self, full_path):

        with open(full_path, newline='') as file:
            reader = csv.reader(file, delimiter=',', quotechar='|')
            next(reader)  # skip header row
            row = next(reader)
            self.fx = float(row[0])
            self.fy = float(row[1])
            self.cx = float(row[2])
            self.cy = float(row[3])
            self.s = float(row[4])
            self.fx_std = float(row[5])
            self.fy_std = float(row[6])
            self.cx_std = float(row[7])
            self.cy_std = float(row[8])
            self.s_std = float(row[9])
            self.rms_reproj_error = float(row[10])
            self.radial_dist_coeffs[0] = float(row[11])
            self.radial_dist_coeffs[1] = float(row[12])
            self.radial_dist_coeffs[2] = float(row[13])
            self.tangential_dist_coeffs[0] = float(row[14])
            self.tangential_dist_coeffs[1] = float(row[15])
            self.radial_dist_coeffs_std[0] = float(row[16])
            self.radial_dist_coeffs_std[1] = float(row[17])
            self.radial_dist_coeffs_std[2] = float(row[18])
            self.tangential_dist_coeffs_std[0] = float(row[19])
            self.tangential_dist_coeffs_std[1] = float(row[20])
            self.width = int(row[21])
            self.height = int(row[22])

    def write_parameters_csv(self, full_path):

        header = ['fx', 'fy', 'cx', 'cy', 's', 'fx_std', 'fy_std', 'cx_std', 'cy_std', 's_std', 'rms_reproj_error',
                  'k1', 'k2', 'k3', 'p1', 'p2', 'k1_std', 'k2_std', 'k3_std', 'p1_std', 'p2_std', 'width', 'height']

        row = (self.fx, self.fy, self.cx, self.cy, self.s, self.fx_std, self.fy_std, self.cx_std, self.cy_std,
               self.s_std, self.rms_reproj_error, self.radial_dist_coeffs[0], self.radial_dist_coeffs[1],
               self.radial_dist_coeffs[2], self.tangential_dist_coeffs[0], self.tangential_dist_coeffs[1],
               self.radial_dist_coeffs_std[0], self.radial_dist_coeffs_std[1], self.radial_dist_coeffs_std[2],
               self.tangential_dist_coeffs_std[0], self.tangential_dist_coeffs_std[1], self.width, self.height)

        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(row)
