import cv2
import cv2.aruco as aruco
import numpy as np

squares_x = 15
squares_y = 12
square_len = 140
marker_len = 100
dict_type = aruco.DICT_4X4_100


def generate_standard_svg():
    # Handle OpenCV version differences in the sandbox
    try:
        dictionary = aruco.getPredefinedDictionary(dict_type)
        board = aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    except AttributeError:
        dictionary = aruco.Dictionary_get(dict_type)
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, dictionary)

    # 1 unit = 1 pixel scale for perfect vectorization
    gen_width = squares_x * square_len
    gen_height = squares_y * square_len

    img = board.generateImage((gen_width, gen_height), marginSize=0)

    # Add margin back
    margin = 50
    img_final = cv2.copyMakeBorder(
        img, top=margin, bottom=margin, left=margin, right=margin,
        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    cv2.imwrite('standard_charuco_board.png', img_final)
    # Meshing algorithm
    h, w = img_final.shape
    visited = np.zeros_like(img_final, dtype=bool)
    rects = []

    for y in range(h):
        for x in range(w):
            if img_final[y, x] == 0 and not visited[y, x]:
                cw = 0
                while x + cw < w and img_final[y, x + cw] == 0 and not visited[y, x + cw]:
                    cw += 1
                ch = 1
                while y + ch < h:
                    if np.all(img_final[y + ch, x:x + cw] == 0) and not np.any(visited[y + ch, x:x + cw]):
                        ch += 1
                    else:
                        break
                visited[y:y + ch, x:x + cw] = True
                rects.append((x, y, cw, ch))

    # SVG Construction
    svg_rects = []
    for (rx, ry, rw, rh) in rects:
        svg_rects.append(f'        <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" />')

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}mm" height="{h}mm">
    <rect width="100%" height="100%" fill="white"/>
    <g fill="black">
{chr(10).join(svg_rects)}
    </g>
</svg>'''

    filename = "standard_charuco_board.svg"
    with open(filename, "w") as f:
        f.write(svg_content)
    print(f"[{filename} generated]")


generate_standard_svg()