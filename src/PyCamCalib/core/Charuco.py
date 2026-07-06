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