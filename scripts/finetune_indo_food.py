"""
Fine-tune YOLOv8n deteksi makanan multi-lauk (NutriSafe).
- Pretrained: model Indonesia (wuriyanto/yolo8-indonesian-food-detection-v1) yg sudah didownload -> datasets/indo_food_yolo/model.pt
- Dataset: SohlHealth/sohl-multidish-yolo-dataset (377 piring multi-lauk, 16 kelas YOLO)
- Map 16 kelas -> kode gizi NutriSafe agar output langsung usable.
- CPU train (no GPU), imgsz 416, epoch rendah biar cepat.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(r"D:/BUAT MBG/datasets/indo_food_yolo")
DATASET_LOCAL = ROOT / "sohl_multidish"
DS_REPO = "SohlHealth/sohl-multidish-yolo-dataset"

# 16 kelas asli SohlHealth -> kode gizi NutriSafe
CLASS_MAP = {
    0: "ROTI_TAWAR",        # bread_or_Roti_naan
    1: "SAYUR_SOP",         # curry_dish
    2: "NASI_PUTIH",        # rice_dish
    3: "TUMIS_SAYUR_HIJAU", # dry_vegetable
    4: "KUE_DONAT",         # snack_item
    5: "KUE_DONAT",         # sweet_item
    6: "SAMBAL_TERASI",     # accompaniment
    7: "SAYUR_SOP",         # Dal_or_sambar
    8: "SAYUR_SOP",         # drink (soup-like) -> map ke sup
    9: "TELUR_DADAR",       # eggs
    10: "IKAN_SEAFOOD",      # fish_dish
    11: "APEL",             # fruits
    12: "MIE_GORENG",       # pasta
    13: "TUMIS_SAYUR_HIJAU",# salad
    14: "SAYUR_SOP",        # soup
    15: "ROTI_TAWAR",       # south_indian_breakfast
}
# Kelas unik NutriSafe yang dipakai (urutan untuk data.yaml)
UNIQUE_CODES = [
    "NASI_PUTIH", "ROTI_TAWAR", "SAYUR_SOP", "TUMIS_SAYUR_HIJAU",
    "SAMBAL_TERASI", "KUE_DONAT", "TELUR_DADAR", "IKAN_SEAFOOD",
    "APEL", "MIE_GORENG",
]
CODE_TO_IDX = {c: i for i, c in enumerate(UNIQUE_CODES)}


def remap_labels(src_labels: Path, dst_labels: Path):
    dst_labels.mkdir(parents=True, exist_ok=True)
    for txt in src_labels.rglob("*.txt"):
        out_lines = []
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                old_cls = int(parts[0])
                new_code = CLASS_MAP.get(old_cls)
                if new_code is None:
                    continue
                new_idx = CODE_TO_IDX.get(new_code)
                if new_idx is None:
                    continue
                out_lines.append(f"{new_idx} {' '.join(parts[1:5])}\n")
        # Tulis ke folder labels yg sudah di-remap
        rel = txt.relative_to(src_labels)
        (dst_labels / rel).parent.mkdir(parents=True, exist_ok=True)
        with open(dst_labels / rel, "w") as f:
            f.writelines(out_lines)


def main():
    print("[1/3] Download dataset SohlHealth...")
    snap = snapshot_download(
        repo_id=DS_REPO,
        repo_type="dataset",
        local_dir=str(DATASET_LOCAL),
        local_dir_use_symlinks=False,
    )
    print("   downloaded to", snap)

    src_labels = DATASET_LOCAL / "labels"
    dst_labels = DATASET_LOCAL / "labels_nutrisafe"
    print("[2/3] Remap labels 16-kelas -> 10 kode NutriSafe...")
    remap_labels(src_labels, dst_labels)

    # Tulis data.yaml NutriSafe
    yaml_path = DATASET_LOCAL / "data_nutrisafe.yaml"
    names_yaml = "\n".join(f"  {i}: {c}" for i, c in enumerate(UNIQUE_CODES))
    yaml_path.write_text(
        f"path: {DATASET_LOCAL}\n"
        f"train: images\nval: images\ntest: images\n"
        f"nc: {len(UNIQUE_CODES)}\nnames:\n{names_yaml}\n"
    )
    print("   wrote", yaml_path)

    print("[3/3] Fine-tune YOLOv8n (CPU, imgsz 416, 40 epoch)...")
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "model.pt"))
    model.train(
        data=str(yaml_path),
        epochs=40,
        imgsz=416,
        batch=8,
        pretrained=True,
        optimizer="auto",
        patience=15,
        close_mosaic=10,
        project=str(ROOT / "runs"),
        name="nutrisafe_multidish",
        exist_ok=True,
        device="cpu",
    )
    # Export best -> ONNX
    best = ROOT / "runs" / "nutrisafe_multidish" / "weights" / "best.pt"
    if best.exists():
        m = YOLO(str(best))
        m.export(format="onnx", imgsz=416, dynamic=False, simplify=True, opset=17)
        print("FINETUNE + EXPORT DONE ->", best)
    else:
        print("TRAIN selesai tapi best.pt tidak ditemukan")


if __name__ == "__main__":
    main()
