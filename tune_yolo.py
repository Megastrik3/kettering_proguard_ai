from ultralytics import YOLO
import os
import numpy as np

def getModels():
    existing_models = os.listdir('./trained_models/')
    if existing_models != []:
        num_models = 1
        trained_models = {}
        for model in existing_models:
            trained_models[num_models] = model
            num_models += 1
        while True:
                print("Trained Models:")
                for key, value in trained_models.items():
                    print(f"{key}: {value}")
                model_choice = input(f"Please select a model [1-{len(trained_models)}]: ")
                if int(model_choice) > len(trained_models) or int(model_choice) < 1 or model_choice.isdigit() is False:
                    print(f"Invalid input. Please enter a number between 1 and {len(trained_models)}.")
                    continue
                else:
                    model_choice = int(model_choice)
                    return (f'trained_models/{trained_models[model_choice]}')
                    break
    elif existing_models == []:
        print("Using default model...")
        model_choice = 0
        return (f'trained_models/{trained_models[model_choice]}')
    

def tuneModel(model_name):
    # Load the model
    model = YOLO(model_name)

    # Run the evaluation
    model.tune(
        data="./datasets/bus-aps/data.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=True, save=False, val=True, device=0
    )


if __name__ == "__main__":
    model_name = getModels()
    if model_name:
        tuneModel(model_name)
    else:
        print("No models available for evaluation.")