import shutil
import gpu_verify
from ultralytics import YOLO
import subprocess


def main():
    print("Starting YOLO Training...")
    model = YOLO("yolo11n-seg.pt")  # Load a COCO-pretrained YOLO11n model
    results = model.train(data="datasets/bus-aps/data.yaml", epochs=1, imgsz=640)
    print("Training completed.")
    print(results)
    import model_export
    model_export.export_model(model)


if __name__ == '__main__':
    main()
