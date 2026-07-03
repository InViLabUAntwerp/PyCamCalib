import cv2
import cv2.aruco
import numpy as np


def brute_force_charuco(image_path, square_length=14.0, marker_length=10.0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Test all common dictionaries to see which one detects the most markers
    dictionaries_to_test = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    }

    params = cv2.aruco.DetectorParameters()
    best_dict_name = None
    best_dict_obj = None
    best_marker_corners = None
    best_marker_ids = None
    max_markers = 0

    print("--- STEP 1: Finding correct ArUco Dictionary ---")
    for dict_name, dict_id in dictionaries_to_test.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        num_markers = len(ids) if ids is not None else 0
        print(f"Tested {dict_name}: Found {num_markers} markers")

        if num_markers > max_markers:
            max_markers = num_markers
            best_dict_name = dict_name
            best_dict_obj = aruco_dict
            best_marker_corners = corners
            best_marker_ids = ids

    if max_markers < 4:
        print(
            "\nFAILURE: Could not find enough raw ArUco markers. The image might be too blurry, or it uses a custom unsupported dictionary.")
        return

    print(f"\n=> WINNER: {best_dict_name} with {max_markers} markers detected.")

    # 2. Test Grid Dimensions and Legacy Pattern Flags
    print("\n--- STEP 2: Brute-forcing Board Dimensions and Legacy Flags ---")
    print("This might take a few seconds...")

    best_corners_count = 0
    best_w, best_h = 0, 0
    best_legacy_flag = False
    best_charuco_corners = None
    best_charuco_ids = None

    # Test grid widths and heights between 8 and 25
    for w in range(8, 25):
        for h in range(8, 25):
            for legacy_flag in [True, False]:

                board = cv2.aruco.CharucoBoard((w, h), square_length, marker_length, best_dict_obj)

                # Apply legacy pattern if the OpenCV version supports it
                if hasattr(board, 'setLegacyPattern'):
                    board.setLegacyPattern(legacy_flag)

                try:
                    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        best_marker_corners, best_marker_ids, gray, board
                    )

                    num_corners = len(charuco_corners) if (charuco_corners is not None and retval) else 0

                    if num_corners > best_corners_count:
                        best_corners_count = num_corners
                        best_w = w
                        best_h = h
                        best_legacy_flag = legacy_flag
                        best_charuco_corners = charuco_corners
                        best_charuco_ids = charuco_ids
                except Exception:
                    pass  # Ignore OpenCV assertions for wildly incorrect dimension maps

    # 3. Output Results
    if best_corners_count > 0:
        print("\n--- SUCCESS! FOUND A WORKING CONFIGURATION ---")
        print(f"Detected Corners: {best_corners_count}")
        print(f"Dictionary:       {best_dict_name}")
        print(f"Grid Layout:      {best_w} columns (Width) x {best_h} rows (Height)")
        print(f"Legacy Pattern:   {best_legacy_flag}")

        # Draw and display the optimal configuration
        image_copy = image.copy()
        cv2.aruco.drawDetectedCornersCharuco(image_copy, best_charuco_corners, best_charuco_ids)

        window_name = f"Success: {best_w}x{best_h} | {best_dict_name} | Legacy: {best_legacy_flag}"
        cv2.imshow(window_name, cv2.resize(image_copy, (720, 960)))
        print("\nPress any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nFAILURE: Found the markers, but could not interpolate the grid.")
        print(
            "This means the squares_X/squares_Y layout on the plate is entirely custom/irregular, or the square/marker size ratio (14mm / 10mm) is vastly different in reality.")


if __name__ == "__main__":
    brute_force_charuco("test.jpg")