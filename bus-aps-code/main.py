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
            print("Using existing model.")
            exported_model = train.export_model()
            if exported_model is None:
                print("Export failed. Using the best model instead.")
                exported_model = 'best.pt'
            yolo.main(exported_model)
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue

if __name__ == '__main__':
    main()