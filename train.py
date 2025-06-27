import shutil
import gpu_verify
from ultralytics import YOLO
from time import sleep
import subprocess
import time
import datetime

export_formats = {
    1: "torchscript",
    2: "onnx",
    3: "openvino",
    4: "engine",
    5: "coreml",
    6: "saved_model",
    7: "pb",
    8: "tflite",
    9: "edgetpu",
    10: "tfjs",
    11: "paddle",
    12: "mnn",
    13: "ncnn",
    14: "imx",
    15: "rknn"
}


def main():
    print("Starting YOLO Training...")
    model = YOLO("yolo11n-seg.pt")  # Load a COCO-pretrained YOLO11n model
    results = model.train(data="datasets/bus-aps/data.yaml", epochs=200, imgsz=640, batch=0.8, device=0, plots=True, resume=True)
    print("Training completed.")
    print(results)
    try:
        version_code = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        shutil.copy(results, f'latest-seg-{version_code}.pt')
        return f'latest-seg-{version_code}.pt'
    except Exception as e:
        print(f"Error copying file: {e}")
        return 'best-seg.pt'


def export_model(model_name='best-seg.pt'):
    print("Do you want to export the model? (y/n): ", end="")
    do_train = input().strip().lower()
    if do_train == 'n':
        print("Skipping export.")
        return
    while True:
        print("Export options:")
        for key, value in export_formats.items():
            print(f"{key}: {value}")
        export_format = input("Enter the export format [1-15]: ")
        if not export_format.isdigit():
            print("Invalid input. Please enter a number between 1 and 15.")
            continue
        else:
            export_format = int(export_format)
            break
    try:
        if export_formats[export_format] == "engine" and not gpu_verify.check_gpu():
            print("GPU not detected. Cannot export to engine format.")
            raise FileNotFoundError("GPU not detected. Cannot export to engine format.")
        model = YOLO(model_name)  # Load the model
        output = model.export(format=export_formats[export_format])
        print(f"Model exported successfully in {export_formats[export_format]} format.")
        print(output)
        print("Copying model to Tailscale...")
        try:
            subprocess.run(['tailscale', 'file', 'cp', f"{str(model_name).removesuffix('.pt')}.{export_formats[export_format]}", '100.67.90.77:'], shell=True, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error copying file to Tailscale: {e}")
        print("Exporting completed successfully")
        return output
    except FileNotFoundError as nf:
        print(f"Error: {nf}. Please ensure the model file exists.")
    except Exception as e:
        print(f"An error occurred during export: {e}")
        print("Export failed. Please check the format and try again.")
    sleep(2)
    print("Exiting YOLO Model Exporter...")
