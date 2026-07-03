import cv2
import cv2.aruco as aruco
import numpy as np

squares_x = 15
squares_y = 12
square_len = 140
marker_len = 100
dict_type = aruco.DICT_4X4_100
dictionary = aruco.getPredefinedDictionary(dict_type)

gen_width = squares_x * square_len
gen_height = squares_y * square_len

# Start with a black canvas, so we can draw white squares!
img = np.zeros((gen_height, gen_width), dtype=np.uint8)

current_id = 0
margin = (square_len - marker_len) // 2

for y in range(squares_y):
    for x in range(squares_x):
        # (0,0) must be WHITE
        if (x + y) % 2 == 0:
            # Draw white square
            img[y * square_len:(y + 1) * square_len, x * square_len:(x + 1) * square_len] = 255

            # Generate ArUco marker
            marker_img = aruco.generateImageMarker(dictionary, current_id, marker_len)

            # Place marker inside the white square
            y_start = y * square_len + margin
            y_end = y_start + marker_len
            x_start = x * square_len + margin
            x_end = x_start + marker_len

            img[y_start:y_end, x_start:x_end] = marker_img
            current_id += 1

# 1. Save the PNG
cv2.imwrite("test_wenglor.png", img)
print("Markers used:", current_id)
print("Saved test_wenglor.png")

# --- SVG EXPORT CODE ---
print("Meshing image into vector coordinates for SVG...")
h, w = img.shape
visited = np.zeros_like(img, dtype=bool)
rects = []

# Map all black pixels into perfect mathematical rectangles
for y in range(h):
    for x in range(w):
        if img[y, x] == 0 and not visited[y, x]:
            cw = 0
            while x + cw < w and img[y, x + cw] == 0 and not visited[y, x + cw]:
                cw += 1
            ch = 1
            while y + ch < h:
                if np.all(img[y + ch, x:x + cw] == 0) and not np.any(visited[y + ch, x:x + cw]):
                    ch += 1
                else:
                    break
            visited[y:y + ch, x:x + cw] = True
            rects.append((x, y, cw, ch))

svg_rects = []
for (rx, ry, rw, rh) in rects:
    svg_rects.append(f'        <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" />')

svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}mm" height="{h}mm">
    <rect width="100%" height="100%" fill="white"/>

    <g fill="black">
{chr(10).join(svg_rects)}
    </g>
</svg>'''

# 2. Save the SVG
with open("test_wenglor.svg", "w") as f:
    f.write(svg_content)
print("Saved test_wenglor.svg")