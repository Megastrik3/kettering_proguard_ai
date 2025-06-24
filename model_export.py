from ultralytics import YOLO
import shutil
from time import sleep
import subprocess

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

print("Starting YOLO Model Exporter...")
sleep(2)

def export_model(model):
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
        output = model.export(format=export_formats[export_format])
        shutil.copy(output, f"./busaps.{export_formats[export_format]}")
        print(f"Model exported successfully in {export_formats[export_format]} format.")
        subprocess.run(['tailscale', 'file', 'cp', f"./busaps.{export_formats[export_format]}", '100.67.90.77:'f"./busaps.{export_formats[export_format]}"], shell=True, capture_output=True, check=True)
        return output
    except FileNotFoundError as nf:
        print(f"Error: {nf}. Please ensure the model file exists.")
    except Exception as e:
        print(f"An error occurred during export: {e}")
        print("Export failed. Please check the format and try again.")
    sleep(2)
    print("Exiting YOLO Model Exporter...")