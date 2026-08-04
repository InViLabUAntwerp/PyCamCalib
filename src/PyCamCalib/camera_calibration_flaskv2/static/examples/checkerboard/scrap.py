from pathlib import Path
from PIL import Image

folder = Path(r"C:\Users\Seppe\PycharmProjects\calibration_toolbox_python_dev\src\PyCamCalib\test\static\examples\checkerboard")

# compress JPEGs already in the folder
for p in folder.iterdir():
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.save(p, "JPEG", quality=75, optimize=True)  # tweak quality (e.g., 70-85)
