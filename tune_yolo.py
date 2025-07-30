from ultralytics import YOLO
import os
import numpy as np
import main

def tuneModel(model_name):
    # Load the model
    model = YOLO(model_name)

    # Run the evaluation
    model.tune(
        data="./datasets/bus-aps/data.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False, device=0
    )


if __name__ == "__main__":
    model_name = main.getModels()
    if model_name:
        tuneModel(model_name)
    else:
        print("No models available for evaluation.")