import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional




# ==========================================
# FEATURE-DETECTION COMPATIBILITY HELPERS
# ==========================================
def get_aruco_dict(dict_id):
    if hasattr(aruco, 'getPredefinedDictionary'):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def get_detector_params():
    if hasattr(aruco, 'DetectorParameters'):
        return aruco.DetectorParameters()
    return aruco.DetectorParameters_create()


def create_charuco_board(w, h, square_len, marker_len, aruco_dict):
    if hasattr(aruco, 'CharucoBoard'):
        return aruco.CharucoBoard((w, h), square_len, marker_len, aruco_dict)
    return aruco.CharucoBoard_create(w, h, square_len, marker_len, aruco_dict)


def get_board_dict(board):
    return board.getDictionary() if hasattr(board, 'getDictionary') else board.dictionary


def detect_markers(image, aruco_dict, params):
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(aruco_dict, params)
        return detector.detectMarkers(image)
    return aruco.detectMarkers(image, aruco_dict, parameters=params)


def refine_markers(image, board, corners, ids, rejected, params):
    if ids is None or len(ids) == 0:
        return corners, ids

    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(get_board_dict(board), params)
        res = detector.refineDetectedMarkers(image, board, corners, ids, rejected)
        if res is not None and len(res) >= 2:
            return res[0], res[1]
    else:
        res = aruco.refineDetectedMarkers(image, board, corners, ids, rejected, parameters=params)
        if res is not None and len(res) >= 2:
            return res[0], res[1]
    return corners, ids


def interpolate_corners(corners, ids, image, board):
    if ids is None or len(ids) == 0:
        return 0, None, None

    # We prioritize the legacy function if it exists.
    # It is faster because it reuses the already-detected markers.
    if hasattr(aruco, 'interpolateCornersCharuco'):
        return aruco.interpolateCornersCharuco(corners, ids, image, board)

    # Fallback to the new detector (Required for clean Linux installs)
    elif hasattr(aruco, 'CharucoDetector'):
        charuco_detector = aruco.CharucoDetector(board)
        # Note: ONLY pass the image here.
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(image)
        retval = len(charuco_corners) if charuco_corners is not None else 0
        return retval, charuco_corners, charuco_ids

    return 0, None, None

# ==========================================
# MAIN DETECTOR CLASS
# ==========================================
class CharucoDetector:
    def __init__(self, board: Optional[aruco.CharucoBoard] = None, search_range: Tuple[int, int] = (10, 18)):
        self.board = board

        # Use compatibility helper
        self.params = get_detector_params()
        self.params.adaptiveThreshWinSizeMin = 3
        self.params.adaptiveThreshWinSizeMax = 23
        self.params.minMarkerPerimeterRate = 0.03

        self.search_min, self.search_max = search_range

    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        b_dict = get_board_dict(self.board)
        corners, ids, rejected = detect_markers(gray, b_dict, self.params)

        if ids is not None and len(ids) > 0:
            refined_corners, refined_ids = refine_markers(
                gray, self.board, corners, ids, rejected, self.params
            )
            return refined_ids, refined_corners
        return None, None

    def auto_detect_board(self, image: np.ndarray, square_len: float, marker_len: float, dict_id=aruco.DICT_4X4_100):
        aruco_dict = get_aruco_dict(dict_id)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detect_markers(gray, aruco_dict, self.params)

        if ids is None or len(ids) == 0:
            return None

        best_score = 0
        best_config = None

        for w in range(self.search_min, self.search_max):
            for h in range(self.search_min, self.search_max):
                for legacy in [True, False]:
                    board = create_charuco_board(w, h, square_len, marker_len, aruco_dict)
                    if hasattr(board, 'setLegacyPattern'): board.setLegacyPattern(legacy)

                    retval, c_corners, c_ids = interpolate_corners(corners, ids, gray, board)

                    if retval and c_corners is not None:
                        score = len(c_corners)
                        if score > best_score:
                            best_score = score
                            best_config = (w, h, legacy)
                            self.board = board

        if best_config:
            w, h, legacy = best_config
            print(f"Optimal Board: {w}x{h} | Legacy: {legacy} | Corners Found: {best_score}")
            self.board = create_charuco_board(w, h, square_len, marker_len, aruco_dict)
            if hasattr(self.board, 'setLegacyPattern'): self.board.setLegacyPattern(legacy)

        return self.board

    def plot_detections(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, rejected = detect_markers(gray, get_board_dict(self.board), self.params)

        if marker_ids is not None and len(marker_ids) > 0:
            ret, corners, ids = interpolate_corners(marker_corners, marker_ids, gray, self.board)
        else:
            ret, corners, ids = False, None, None

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if ret and corners is not None:
            ax.scatter(corners[:, 0, 0], corners[:, 0, 1], c='red', s=8, label='Corners')
            for i, idx in enumerate(ids):
                ax.text(corners[i, 0, 0], corners[i, 0, 1], str(idx[0]), color='yellow', fontsize=7)
        plt.show()


if __name__ == "__main__":
    # Add a fallback just so the script doesn't crash if test.jpg is missing
    img = cv2.imread("test.jpg")
    if img is None:
        print("Image 'test.jpg' not found. Creating a blank image for script execution...")
        img = np.zeros((800, 800, 3), dtype=np.uint8)

    # --- EXAMPLE 1: AUTO-DETECTION ---
    print("--- Example 1: Auto-detection ---")
    detector_auto = CharucoDetector(search_range=(10, 18))
    detector_auto.auto_detect_board(img, square_len=14.0, marker_len=10.0)
    if detector_auto.board:
        detector_auto.plot_detections(img)

    # --- EXAMPLE 2: KNOWN PARAMETERS ---
    print("\n--- Example 2: Known Parameters (15x12) ---")
    known_dict = get_aruco_dict(aruco.DICT_4X4_100)
    known_board = create_charuco_board(15, 12, 14.0, 10.0, known_dict)
    if hasattr(known_board, 'setLegacyPattern'):
        known_board.setLegacyPattern(True)

    detector_known = CharucoDetector(board=known_board)
    detector_known.plot_detections(img)