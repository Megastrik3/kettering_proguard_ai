import os
from time import sleep
import train
import yolo

def main():
    print("Starting bus anti-pinch system...")
    sleep(2)
    print("Checking for new model...")
    ask_for_traning()



def ask_for_traning():
    while True:
        ans = input("Do you want to train a new model? (y/n): ").strip().lower()
        if ans == 'y':
            model_output = train.main()
            exported_model = train.export_model(model_output)
            if exported_model is None:
                print("Export failed. Using the best model instead.")
                exported_model = 'best.pt'
            yolo.main(exported_model)
            break
        elif ans == 'n':
            print("Using existing model...")
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
                        if not model_choice.isdigit():
                            print(f"Invalid input. Please enter a number between 1 and {len(trained_models)}.")
                            continue
                        else:
                            model_choice = int(model_choice)
                            exported_model = train.export_model(f'trained_models/{trained_models[model_choice]}')
                            yolo.main(exported_model)
                            break
            elif existing_models == []:
                print("Using default model...")
                model_choice = 0
                exported_model = train.export_model("yolo11n.pt")
                yolo.main(exported_model)
                break
            if exported_model is None:
                print("Export failed. Using the best model instead.")
                exported_model = 'yolo11n.pt'
            yolo.main(exported_model)
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue

if __name__ == '__main__':
    main()