import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

REQUIRED_CLASSES = {
    "ikan_bakar",
    "ikan_goreng",
    "ayam_goreng",
    "tempe",
    "tahu",
    "telur",
    "nasi_putih",
    "sayur",
    "sambal",
}


def get_class_names(data: dict) -> set[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return {str(name).strip().lower() for name in names.values()}
    if isinstance(names, list):
        return {str(name).strip().lower() for name in names}
    raise ValueError("Field 'names' pada data.yaml harus berupa daftar atau mapping kelas YOLO.")


def validate_split(dataset_root: Path, split: str) -> int:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Split '{split}' harus memiliki images/{split} dan labels/{split}.")

    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        raise ValueError(f"Tidak ada foto nyata pada images/{split}.")

    missing_labels = [image for image in images if not (label_dir / f"{image.stem}.txt").is_file()]
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} foto pada images/{split} tidak memiliki anotasi bounding-box YOLO.")

    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih YOLO11 hanya dari dataset makanan nyata yang telah dianotasi.")
    parser.add_argument("--data", required=True, type=Path, help="Lokasi data.yaml dataset YOLO.")
    parser.add_argument("--model", default="yolo11s.pt", help="Checkpoint YOLO11 Ultralytics.")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--project", default="runs/nutrisafe")
    parser.add_argument("--name", default="mbg-real-food")
    args = parser.parse_args()

    data_file = args.data.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {data_file}")

    dataset = YAML.load(str(data_file))
    dataset_root = Path(dataset.get("path") or data_file.parent).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_file.parent / dataset_root).resolve()

    class_names = get_class_names(dataset)
    missing_classes = REQUIRED_CLASSES - class_names
    if missing_classes:
        raise ValueError("Dataset belum memuat kelas wajib MBG: " + ", ".join(sorted(missing_classes)))

    train_count = validate_split(dataset_root, "train")
    val_count = validate_split(dataset_root, "val")
    if train_count < 200 or val_count < 50:
        raise ValueError("Dataset terlalu kecil. Gunakan minimal 200 foto latih dan 50 foto validasi yang dianotasi manusia.")

    model = YOLO(args.model)
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="auto",
        patience=25,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
