import h5py

def return_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        calibration = h5_file["camera_calibration/camera_parameters"]
        print(calibration.keys())
        c = calibration["c"]
        print(c[:])
        f = calibration["f"]
        print(f[:])
        rad_dst = calibration["radial_dist_coeffs"]
        print(rad_dst[:])
        tan_dst = calibration["tangential_dist_coeffs"]
        print(tan_dst[:])
        cam_in = [
            f[0], 0, c[0],
            0, f[1], c[1],
            0, 0, 1
        ]
        cam_dist = [rad_dst[0], rad_dst[1], tan_dst[0], tan_dst[1], rad_dst[2]]
        print("c_std")
        print(calibration["c_std"][:])
        print("f_std")
        print(calibration["f_std"][:])

h5_file_path = r'calib/camera_calibration/camera_1_parameters.h5'  # Replace with your H5 file path
return_data(h5_file_path)
