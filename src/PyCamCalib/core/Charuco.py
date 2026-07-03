import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class CharucoDetector:
    def __init__(self, board: Optional[aruco.CharucoBoard] = None, search_range: Tuple[int, int] = (10, 18)):
        self.board = board
        self.params = aruco.DetectorParameters()
        self.params.adaptiveThreshWinSizeMin = 3
        self.params.adaptiveThreshWinSizeMax = 23
        self.params.minMarkerPerimeterRate = 0.03

        # Configure search range as a class parameter
        self.search_min, self.search_max = search_range

    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corners, ids, rejected = aruco.detectMarkers(gray, self.board.getDictionary(), parameters=self.params)

        if ids is not None:
            refined_corners, refined_ids, _, _ = aruco.refineDetectedMarkers(
                gray, self.board, corners, ids, rejected, parameters=self.params
            )
            return refined_ids, refined_corners
        return None, None

    def auto_detect_board(self, image: np.ndarray, square_len: float, marker_len: float, dict_id=aruco.DICT_4X4_100):
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=self.params)

        if ids is None:
            return None

        best_score = 0
        best_config = None

        # Uses the configured search range
        for w in range(self.search_min, self.search_max):
            for h in range(self.search_min, self.search_max):
                for legacy in [True, False]:
                    board = aruco.CharucoBoard((w, h), square_len, marker_len, aruco_dict)
                    if hasattr(board, 'setLegacyPattern'): board.setLegacyPattern(legacy)

                    retval, c_corners, c_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

                    if retval and c_corners is not None:
                        score = len(c_corners)
                        if score > best_score:
                            best_score = score
                            best_config = (w, h, legacy)
                            self.board = board

        if best_config:
            w, h, legacy = best_config
            print(f"Optimal Board: {w}x{h} | Legacy: {legacy} | Corners Found: {best_score}")
            self.board = aruco.CharucoBoard((w, h), square_len, marker_len, aruco_dict)
            if hasattr(self.board, 'setLegacyPattern'): self.board.setLegacyPattern(legacy)

        return self.board

    def plot_detections(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, self.board.getDictionary(),
                                                                   parameters=self.params)

        if marker_ids is not None:
            ret, corners, ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, self.board)
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
    img = cv2.imread("test.jpg")

    # --- EXAMPLE 1: AUTO-DETECTION ---
    print("--- Example 1: Auto-detection ---")
    detector_auto = CharucoDetector(search_range=(10, 18))
    detector_auto.auto_detect_board(img, square_len=14.0, marker_len=10.0)
    if detector_auto.board:
        detector_auto.plot_detections(img)

    # --- EXAMPLE 2: KNOWN PARAMETERS ---
    print("\n--- Example 2: Known Parameters (15x12) ---")

    # Create the board manually using standard ArUco dictionary
    known_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    known_board = aruco.CharucoBoard((15, 12), 14.0, 10.0, known_dict)
    if hasattr(known_board, 'setLegacyPattern'):
        known_board.setLegacyPattern(True)

    # Initialize detector with the known board
    detector_known = CharucoDetector(board=known_board)

    # Plotting the known board without needing any cv2 calls in main
    detector_known.plot_detections(img)