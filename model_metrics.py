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
    

def generateMetrics(model_name):
    # Load the model
    model = YOLO(model_name)

    # Run the evaluation
    results = model.val(data="./datasets/bus-aps/data.yaml")

    # Print specific metrics
    import matplotlib.pyplot as plt

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    for i, (p_curve, r_curve) in enumerate(zip(results.box.p_curve, results.box.r_curve)):
        plt.plot(r_curve, p_curve, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot F1 score curve
    plt.figure(figsize=(8, 6))
    for i, f1_curve in enumerate(results.box.f1_curve):
        plt.plot(np.linspace(0, 1, len(f1_curve)), f1_curve, label=f'Class {i}')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot mAP at different IoU thresholds
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.5, 0.95, len(results.box.maps)), results.box.maps, marker='o')
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP) at Different IoU Thresholds')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    model_name = getModels()
    if model_name:
        generateMetrics(model_name)
    else:
        print("No models available for evaluation.")