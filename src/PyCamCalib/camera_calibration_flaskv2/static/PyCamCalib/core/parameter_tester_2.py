import cv2
import cv2.aruco as aruco
import numpy as np


def analyze_wenglor_plate(image_path="test.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # We know this is the winning dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    params = aruco.DetectorParameters()

    # Detect raw markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is not None:
        print(f"Total markers found: {len(ids)}")
        print(f"Minimum ID: {np.min(ids)} | Maximum ID: {np.max(ids)}")

        drawn_img = image.copy()

        # Draw just the square borders first
        aruco.drawDetectedMarkers(drawn_img, corners)

        # Manually draw large IDs in the center of each marker
        for i in range(len(ids)):
            marker_id = str(ids[i][0])
            marker_corners = corners[i][0]

            # Calculate the center of the marker
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))

            # Draw the text! Change fontScale and thickness here if you need to
            cv2.putText(
                drawn_img,
                marker_id,
                (center_x - 20, center_y + 15),  # Shift slightly to center the text
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(0, 0, 255),  # Red color (B, G, R)
                thickness=4,
                lineType=cv2.LINE_AA
            )

        # Display the image
        cv2.imshow("Wenglor ID Pattern Inspection", cv2.resize(drawn_img, (720, 960)))
        print("\nLook closely at the image window.")
        print("Where is the lowest ID located? Does it increase going right, or going down?")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No markers found.")


if __name__ == "__main__":
    analyze_wenglor_plate()