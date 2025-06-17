import shutil
import gpu_verify
from ultralytics import YOLO


def main():
    print("Starting YOLO Training...")
    model = YOLO("yolo11n.pt")  # Load a COCO-pretrained YOLO11n model
    results = model.train(data="datasets/bus-aps/data.yaml", epochs=100, imgsz=640)
    print("Training completed.")
    print(results)
    if gpu_verify.check_gpu():
        model_format = "engine"
    else:
        model_format = "onnx"

    output = model.export(format=model_format)
    shutil.copy(output, f"./busaps.{model_format}")


if __name__ == '__main__':
    main()
