# ProGuard: A Preemptive Real-time Artificial Intelligence-Based Anti-Pinch System for Bus Entryways on Low-Power Edge Devices
<p>This paper proposes ProGuard, a novel approach to preemptive pinch detection systems for buses. ProGuard utilizes state-of-the-art AI object detection algorithms to identify potential pinching events in bus entryways before pinching occurs. Modern conventional anti-pinch systems, such as pressure sensors or hall effect sensors, often rely on mechanical contact before triggering. While these systems are established safety mechanisms, they are reactive and therefore require some level of pinching before triggering. This reactive approach presents numerous safety concerns for passengers, especially when considering children on school buses. Existing preemptive detection methods, such as infrared or ultrasonic sensors, solve the problems presented by these reactive detection systems. However, these systems either lack the range or environmental resilience needed for reliable operation in buses. The critical nature of anti-pinch systems requires a robust and reliable solution that can adapt to various applications and environments. Our study investigates an AI-based approach that leverages the YOLOv11 nano object detection model to detect people and backpacks in real-time. We performed a comparative study on various model formats to find the best-performing format on the chosen edge compute hardware. Our experimental results revealed that when using the IMX model format on an AI-accelerated camera, ProGuard can achieve 24 frames per second and an inference time of 125ms while running on a Raspberry Pi computer. Performance tests on this model showed a mAP@0.5-0.95 of 0.522, putting ProGuard on par with baseline YOLOv11 nano performance. These results demonstrate that ProGuard offers an efficient and real-time alternative to current pinch detection approaches while operating on low-cost consumer hardware. <p/>


<a href="https://universe.roboflow.com/hudson-bradley/proguard-fsphe">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

<a href="https://universe.roboflow.com/hudson-bradley/proguard-fsphe/model/">
    <img src="https://app.roboflow.com/images/try-model-badge.svg"></img>
</a>

# Repository Map
The repository is structured as such:

### Python Scripts
- `main.py`: This is the base script that calls train.py, gpu_verfy.py, sonyimx500.py, and yolo.py. This script coordinates training, exporting, and predicting using a live webcam.
- `capture_data.py`: This script is used to capture images and video on a Rasbperry Pi camera. 
- `model_testing.py`: This script is used to calculate model metrics against the testing dataset.
- `predict.py`: This script is used to run predictions on videos or image directories.
- `sonyimx500.py`: This script is used to upload a model exported in the imx format to a Sony IMX500 camera [(Raspberry Pi AI Camera)](https://www.raspberrypi.com/products/ai-camera/).
- `tune_yolo.py`: This script is used to tune YOLO models.
- `oi_download.py`: This script is used to download a subset of the images from the Open Images dataset.
- `coco_download.py`: This script is used to download a subset of the image from the COCO-2017 dataset.

### Folders
- `/datasets`: this folder contains the active dataset that is being used to train the model. The active dataset resides in a subfolder called "bus-aps".
- `/conda_environment`: This folder holds the conda config file which can be used to load the libraries for this repository.
- `/runs`: This folder contains every version of the trained models and their export varients.

# How to use the Repository

### Download the ProGuard Dataset
1. To download the ProGuard dataset, go to the [Roboflow Universe](https://universe.roboflow.com/hudson-bradley/proguard-fsphe) listing and download the latest version of the dataset in the YOLOv11 format.
> [!TIP]
> - If you are using windows, use the "Download zip to computer" option to download the dataset as a `.zip` folder and extract the contents to the `/datasets` folder.
> - If you are using Linux or MacOS use the "Show download code" option to download the dataset using `curl` to the `/datasets/bus-aps` folder.
2. If using Windows, extract the contents of the `.zip` folder to the `/datasets` directory.
> [!IMPORTANT]
> The dataset files (train, test, valid, etc...) MUST be in a folder in the `/datasets` directory called `bus-aps`. All scripts which use the dataset will look in this dirctory, so it is very important that the dataset be in this folder.

## Training, exporting and testing a model

### Model Training
1. To train ProGuard using the downloaded dataset, run the `main.py` script using the following command:
```bash
python main.py
```
2. After starting this script, it will ask if you want to train a new model version. Entry `y` when prompted and the training script will start automatically.
>[!TIP]
> If you don't have a GPU, the script will fail because it requires a GPU in order to train the model. If you wish to continue without a GPU, please see the `train.py` script for instructions on how to disable required GPU support.
3. Once training has finished, the training output will be saved to `/runs/detect/trainXX`. The best version of the model will be moved to the `/trained_models` folder and will be given the name `latest_[timestamp].pt`.

>[!NOTE]
> ### Default Training Parameters
>- Epochs: 500
>- Image size: 640
>- Batch size: (auto)
>- GPU enabled: True
>- Hue augmentation: 0.25
>- Saturation augmentation: 0.25
>- Exposure augmentation: 0.5
>- Translation augmentaiton: 25%
>- Scale augmentation: 30%
>- Flip (L/R) augmentation: 20%
>- IoU: 70%

### Model Exporting
> [!IMPORTANT]
> If you are going to use the `imx` format, you MUST uncomment the lines referencing the `sonyimx500` package in the `main.py` script. Please see comments in `main.py` for which lines should be uncommented. These lines are commented out because the packages used are only available on the Raspberry Pi (or linux) and therefore do not work on Windows which was the primary development platform.
1. Once the model has finished training you will be asked if you would like to export your newly trained model in a different format. If you chose not to train a new model, you will first be asked to select a model from the `/trained_models` directory before being asked if you would like to export the model in a new format.
2. If you select `y`, a list of all valid export options will be displayed. Use the numbers listed next to the export formats to chose a model format.
> [!TIP]
> Use the `ONNX` format when running in a CPU only environmet. Use the `engine` format when using the model in a GPU ready environment. Use the `imx` format to quantize and export the model for use on the Sony IMX500. The `ncnn` format is used when running the model on a Raspberry Pi with no accelerator.

> [!IMPORTANT]
> The `imx` and `ncnn` model versions can only be exported on a Raspberry Pi (or linux) due to the required libraries. Please see the [Ultralytics IMX500 Docs](https://docs.ultralytics.com/integrations/sony-imx500/#using-imx500-export-in-deployment) for instructions on which packages are required to export the model to the `imx` format. The `engine` format requires a GPU and will fail if a GPU is not present.
3. Once the model has finished exporting, it will be saved in the `/trained_models` directory with the same name but different extension as the originally selected model.

### Live Webcam Testing
1. After exporting the model (or skipping that step), the script will automatically start the live detection process. After the model is succesfully loaded, a window will appear showing the output of the model's inference on the video stream. FPS will be displayed in the corner and detection class and confidence will be displayed on the bounding boxes.
>[!IMPORTANT]
> If a model with the `IMX` format was selected, the script will automatically decompress and upload the model to the Sony IMX500 camera. You may be asked if you want to re-extract the model if you are running the same model version more than one. Enter `n` or just press the `Enter` key to skip unless you suspect an issue with the existing version.
2. To stop detection press the `q` key in the detection window. 
3. After quiting the detection window, the terminal will show the average inference and fps of the model. This capturing process is started 150 frames after the video stream has started in order to give the model time to 'warm up'.
> [!NOTE]
> If using the `imx` version of the model, the script will open a different window to display the model inference output. This window will display FPS and DPS. To close the window, press the `esc` key. The terminal will show the average FPS and DPS in the output.


## Gather data to build a dataset

### Download part of the Open-Images dataset
To download images from the Open-Images dataset, use the `oi_download.py` script. Currently, this script is designed to download no more than 2,000 images from Open-Images from the classes 'person' and 'backpack'. The split that images are pulled from will need to be manually changed according to the user's need. Images are exported to the `/datasets/` folder for easy access. Use the following command:
```bash
python oi_download.py
```

### Download part of the COCO-2017 dataset
To download images from the COCO-2017 dataset, use the `coco_download.py` script. This script will download no more than 2000 images which contain the classes 'person' or 'backpack'. Images are downloaded and exported to the `/datasets/` folder for easy access. Use the following command:
```bash
python coco_download.py
```

### Capturing data from a Raspberry Pi camera
To capture images directly from a Raspberry Pi with attached ribbon cable camera, use the `capture-data.py` script. This script will ask how many images should be captured, then capture that many images with a 1 second delay. Images are saved to the `/data-capture` folder. Once image capture is complete, the script will ask if you want to capture a video as well. Enter `0` to end the script, or the number of seconds of video you want to capture. Recorded videos are also saved to the `/data-capture/` folder. Using the following command:
```bash
python capture-data.py
```



