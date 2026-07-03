from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from PyCamCalib.core.Charuco import CharucoDetector  # Import your CharucoDetector class
from PyCamCalib.core.feature_detection import FeatureDetector, CalibrationFeature  # Import original base class
import cv2 as cv2
import cv2.aruco as aruco

class CharucoFeatureDetector(FeatureDetector):
    # Registry of standard boards
    PRESETS = {
        "wenglor": {
            "size": (15, 12),
            "square_len": 14.0,
            "marker_len": 10.0,
            "dict": aruco.DICT_4X4_100,
            "legacy": True
        },
        # You can add more here later:
        # "small_test": {"size": (5, 5), ...}
    }

    @classmethod
    def get_preset_board(cls, name: str) -> aruco.CharucoBoard:
        cfg = cls.PRESETS[name]
        aruco_dict = aruco.getPredefinedDictionary(cfg["dict"])
        board = aruco.CharucoBoard(cfg["size"], cfg["square_len"], cfg["marker_len"], aruco_dict)
        if hasattr(board, 'setLegacyPattern'):
            board.setLegacyPattern(cfg["legacy"])
        return board

    def __init__(self, space_between_features: float, preset: Optional[str] = None,
                 board: Optional[aruco.CharucoBoard] = None, **kwargs) -> None:
        super().__init__(space_between_features, board_size=(0, 0), **kwargs)
        final_board = board
        if final_board is None and preset is not None:
            final_board = self.get_preset_board(preset)
        self.charuco_detector = CharucoDetector(board=final_board)
        self.use_charuco = True
        self.auto_detected_board = None # To hold the board if auto-detected

    def detect_feature(self, image: npt.NDArray) -> CalibrationFeature:
        # 1. Ensure we have a board
        if self.charuco_detector.board is None and self.auto_detected_board is None:
            marker_len_guess = self.space_between_features * 0.7
            self.auto_detected_board = self.charuco_detector.auto_detect_board(
                image,
                square_len=self.space_between_features,
                marker_len=marker_len_guess
            )
            if self.auto_detected_board is None:
                return CalibrationFeature(0, np.array([]), np.array([]))

        board_to_use = self.auto_detected_board if self.auto_detected_board is not None else self.charuco_detector.board

        # 2. Marker detection
        marker_ids, marker_corners = self.charuco_detector.detect_aruco_markers(image)

        # 3. Interpolate corners
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if marker_ids is not None and len(marker_ids) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board_to_use
            )

            if retval > 0 and charuco_corners is not None and len(charuco_ids) > 0:
                # MANDATORY FORMAT FIX: Ensure float32 and shape (N, 2)
                image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

                # Fetch board object points
                all_obj_points = board_to_use.getChessboardCorners()
                # Use flatten() to ensure indexing works regardless of array shape
                object_points_3d = all_obj_points[charuco_ids.flatten()]
                if object_points_3d.shape[1] == 4:
                    object_points_3d = object_points_3d[:, :3]
                if object_points_3d.shape[1] == 2:
                    zeros = np.zeros((object_points_3d.shape[0], 1), dtype=np.float32)
                    object_points_3d = np.hstack((object_points_3d, zeros))

                # 4. Final cast
                object_points = object_points_3d.astype(np.float32)


                # Scale by square size: not needed, aruco fixes this: 0705 double check!

                return CalibrationFeature(
                    score=2,
                    image_points=image_points,
                    object_points=object_points
                )

        return CalibrationFeature(0, np.array([]), np.array([]))


def extract_and_print_board():
    image = cv2.imread("test.jpg")  # Use your best image
    detector = CharucoFeatureDetector(space_between_features=14.0)

    # 1. Force the auto-detection
    board = detector.charuco_detector.auto_detect_board(
        image,
        square_len=14.0,
        marker_len=10.0  # Standard guess
    )

    if board:
        # 2. Extract parameters
        # Note: Depending on your OpenCV version, these might be attributes or methods
        print("--- BOARD FOUND ---")
        print(f"Chessboard Size: {board.getChessboardSize()}")
        print(f"Square Length: {board.getSquareLength()}")
        print(f"Marker Length: {board.getMarkerLength()}")

        # Dictionary info
        d = board.getDictionary()
        print(f"Dictionary Type: {d.bytesList.shape} bits")

        # Legacy Pattern check (try both common patterns)
        print("Check your board for a white/black square at 0,0 and update 'legacy' accordingly.")
    else:
        print("Board auto-detection failed.")
def test_charuco_pipeline():
    # 1. Load your test image
    image_path = "test.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # 2. Initialize the detector
    # We pass None for the board to trigger the Auto-Detector
    print("Initializing CharucoFeatureDetector...")
    detector = CharucoFeatureDetector(space_between_features=14.0)

    # 3. Detect features
    # This call will trigger auto_detect_board internally on the first run
    print("Running detection...")
    feature = detector.detect_feature(image)

    # 4. Verify results
    if feature.score > 0:
        print(f"Success! Detected {len(feature.image_points)} corners.")
        print(f"Score: {feature.score}")
        print(f"Image points shape: {feature.image_points.shape}")
        print(f"Object points shape: {feature.object_points.shape}")

        # Optional: Visualize the detected points
        debug_img = image.copy()
        for pt in feature.image_points:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        cv2.imshow("Detected Points", cv2.resize(debug_img, (720, 960)))
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Detection failed: Score is 0.")


if __name__ == "__main__":
    test_charuco_pipeline()
    extract_and_print_board()