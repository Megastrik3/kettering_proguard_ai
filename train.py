import shutil
import gpu_verify
from ultralytics import YOLO
from time import sleep
import datetime
import os

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
    model = YOLO("yolo11n.pt")  # Load a COCO-pretrained YOLO11n model
    results = model.train(data="datasets/bus-aps/data.yaml", epochs=450, imgsz=640, batch=-1, device=0, patience=50) # Define the training parameters
    print("Training completed.")
    try:
        version_code = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        training_runs = len(next(os.walk('./runs/detect/'))[1])
        print(f"Number of training runs: {training_runs}")
        if training_runs == 1:
            training_runs = ''
        newest_model = f'runs/detect/train{training_runs}/weights/best.pt'
        shutil.copy(newest_model, f'./trained_models/latest-{version_code}.pt')
        return f'./trained_models/latest-{version_code}.pt'
    except FileNotFoundError as e:
        print(f"Error copying file: {e}")
        return './trained_models/best.pt'
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return './trained_models/best.pt'

def export_model(model_name='best.pt'):
    print("Do you want to export the model? (y/n): ", end="")
    while True:
        do_train = input().strip().lower()
        if do_train == 'n':
            print("Skipping export.")
            return model_name
        elif do_train != 'y':
            print("Invalid input. Please enter 'y' or 'n'.")
            continue
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
            raise Exception("GPU not detected. Cannot export to engine format.")
        model = YOLO(model_name)  # Load the model
        output = model.export(format=export_formats[export_format], name=model_name)
        print(f"Model exported successfully in {export_formats[export_format]} format.")
        print(output)
        print("Exporting completed successfully")
        return output
    except FileNotFoundError as nf:
        print(f"Error: {nf}. Please ensure the model file exists.")
        return model_name
    except Exception as e:
        print(f"An error occurred during export: {e}")
        print("Export failed. Please check the format and try again.")
        return model_name

        
    sleep(2)
    print("Exiting YOLO Model Exporter...")
