"""Export model.pt (wuriyanto yolo8 Indonesian food, 12 kelas: ADA Ayam Goreng,
Tahu, Tempe, dll) -> ONNX untuk ditambahkan ke kandidat YOLO.
Model ini sudah terlatih detect lauk MBG lengkap, jadi gratis tambah ayam/tahu/tempe.
"""
from pathlib import Path

ROOT = Path(r"D:/BUAT MBG/datasets/indo_food_yolo")
SRC = ROOT / "model.pt"
DST = Path(r"D:/BUAT MBG/public/models/yolov8_indo12.onnx")

from ultralytics import YOLO
m = YOLO(str(SRC))
# yolo8 original biasanya imgsz 640
m.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17)
print("exported ->", DST)
# ultralytics menaruh di samping src dgn nama model.onnx
import shutil
auto = SRC.with_suffix(".onnx")
if auto.exists() and not DST.exists():
    shutil.move(str(auto), str(DST))
    print("moved to", DST)
