"""
This script is the main entry point for training, exporting, and testing a YOLO model.
This script will prompt the user to train, select a model, and export the model before running the live inference.
"""
import os
from time import sleep
import train
import yolo
#import sonyimx500

def main():
    print("Starting ProGuard anti-pinch system...")
    sleep(2)
    print("Checking for new model...")
    ask_for_traning()


"""
Prompt the user to train a new model or select an existing one.
Once a model has been trained or selected, detection will be automatically started.
"""
def ask_for_traning():
    while True:
        ans = input("Do you want to train a new model? (y/n): ").strip().lower()
        if ans == 'y':
            model_output = train.main()
            exported_model = train.export_model(model_output)
            if exported_model is None:
                print("Export failed. Using the best model instead.")
                exported_model = 'best.pt'
            if exported_model.find('imx') != -1:
                print("IMX500 model detected. Using IMX500 deployment script.")
               # sonyimx500.main(exported_model)
            else:
                yolo.main(exported_model)
            break
        elif ans == 'n':
            selected_model = getModels()
            if selected_model is not None:
                exported_model = train.export_model(selected_model)
            if exported_model.find('imx') != -1:
                print("IMX500 model detected. Using IMX500 deployment script.")
             #   sonyimx500.main(exported_model)
            else:
                yolo.main(exported_model)
                break
            if exported_model is None:
                print("Export failed. Using the best model instead.")
                exported_model = 'yolo11n.pt'

            if exported_model.find('imx') != -1:
                print("IMX500 model detected. Using IMX500 deployment script.")
             #   sonyimx500.main(exported_model)
            else:
                yolo.main(exported_model)
                break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue


"""
Read the contents of the trained_models directory and return a list of available models.
If no models are found, return the default model 'yolo11n.pt'.
"""
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
        return "yolo11n.pt"


if __name__ == '__main__':
    main()


