import h5py

def return_stereo_parameters(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        stereo_path = "camera_calibration/stereo_parameters"

        if stereo_path not in h5_file:
            print(f"No stereo parameters found at {stereo_path}")
            return

        stereo = h5_file[stereo_path]

        print("\nStereo Parameters:")
        print("RMS Reprojection Error:", stereo["rms_reproj_error"][()])
        print("Rotation Matrix (R):\n", stereo["R"][:])
        print("Translation Vector (T):\n", stereo["T"][:])
        print("Essential Matrix (E):\n", stereo["E"][:])
        print("Fundamental Matrix (F):\n", stereo["F"][:])

        # Transformation Matrix
        if "TransformationMatrix" in stereo:
            tm_group = stereo["TransformationMatrix"]
            print("Homography Matrix (H):\n", tm_group["H"][:])
            print("Info:", tm_group.attrs.get("info", "N/A"))
            print("Units:", tm_group.attrs.get("units", "N/A"))
        else:
            print("No TransformationMatrix group found.")

h5_file_path = r'calib/example_stereo_parameters.h5'
return_stereo_parameters(h5_file_path)
