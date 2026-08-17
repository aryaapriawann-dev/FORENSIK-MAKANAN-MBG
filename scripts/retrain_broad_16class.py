"""
Retrain YOLOv8n NutriSafe — BROAD / banyak makanan, murni dari dataset.
TIDAK ada REQUIRED_CLASSES hardcoded (tidak menyempitkan ke 3 makanan).
Ambil SEMUA 16 kelas asli sohl-multidish:
  0 bread_or_Roti_naan, 1 curry_dish, 2 rice_dish, 3 dry_vegetable,
  4 snack_item, 5 sweet_item, 6 accompaniment(sambal), 7 Dal_or_sambar,
  8 drink, 9 eggs, 10 fish_dish(ikan), 11 fruits(jeruk), 12 pasta(mie),
  13 salad, 14 soup, 15 south_indian_breakfast
Augmentasi agresif agar kelas dengan sampel sedang (ikan/sambal/fruits)
tetap ke-detect dengan yakin. Transfer dari model.pt (wuriyanto) sebagai
pretrained backbone.
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"D:/BUAT MBG/datasets/indo_food_yolo")
SRC = ROOT / "model.pt"          # backbone pretrained (12-kelas Indo)
DATA = ROOT / "data_broad16.yaml"   # 16 kelas asli, path absolut
OUT = ROOT / "runs" / "nutrisafe_broad16"

def main():
    print("[1/2] Loading backbone:", SRC)
    model = YOLO(str(SRC))
    print("[2/2] Training BROAD 16-kelas (augmentasi tinggi)...")
    model.train(
        data=str(DATA),
        task="detect",
        imgsz=416,
        epochs=120,
        batch=8,
        patience=25,
        optimizer="auto",
        pretrained=True,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.02, hsv_s=0.9, hsv_v=0.6,     # augmentasi warna ekstrem
        flipud=0.3, fliplr=0.6,
        mosaic=1.0, mixup=0.3, copy_paste=0.3,
        degrees=20.0, translate=0.25, scale=0.7, shear=8.0,
        project=str(ROOT / "runs"),
        name="nutrisafe_broad16",
        exist_ok=True,
        device="cpu",
    )
    best = OUT / "weights" / "best.pt"
    if best.exists():
        # Export ONNX agar bisa dipakai di web (onnxruntime-web).
        m = YOLO(str(best))
        m.export(format="onnx", imgsz=416, dynamic=False, simplify=True, opset=17)
        print("TRAIN + EXPORT DONE ->", best, "&&", best.with_suffix(".onnx"))
    else:
        print("TRAIN selesai tapi best.pt tidak ditemukan di", best)

if __name__ == "__main__":
    main()
