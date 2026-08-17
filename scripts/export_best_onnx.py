"""Export best.pt (fine-tune NutriSafe) -> ONNX untuk diintegrasikan ke app.
Jalan terpisah dari training agar tidak race dengan penulisan best.pt.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:/BUAT MBG/datasets/indo_food_yolo")
SRC_PT = ROOT / "runs" / "nutrisafe_multidish" / "weights" / "best.pt"
SNAP = ROOT / "runs" / "nutrisafe_multidish" / "weights" / "best_freeze.pt"
DST_ONNX = ROOT / "runs" / "nutrisafe_multidish" / "weights" / "best_export.onnx"

# Snapshot agar tidak race dengan training yang masih menulis best.pt
shutil.copyfile(SRC_PT, SNAP)
print(f"[1] snapshot best.pt -> {SNAP} ({SNAP.stat().st_size/1e6:.1f} MB)")

from ultralytics import YOLO
m = YOLO(str(SNAP))
print("[2] exporting ONNX (imgsz 416, dynamic False)...")
m.export(format="onnx", imgsz=416, dynamic=False, simplify=True, opset=17)
print("[3] done. looking for exported onnx...")
for p in [SNAP.with_suffix(".onnx"), DST_ONNX]:
    if p.exists():
        print("   exported:", p, f"{p.stat().st_size/1e6:.1f} MB")
        break
