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
            train.main()
            yolo.main()
            break
        elif ans == 'n':
            print("Using existing model.")
            yolo.main()
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue

if __name__ == '__main__':
    main()