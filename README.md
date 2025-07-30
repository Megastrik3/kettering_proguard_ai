# ProGuard: A Preemptive Real-time Artificial Intelligence-Based Anti-Pinch System for Bus Entryways on Low-Power Edge Devices
Anti-pinch systems have been a topic of study for many years. Many of these systems rely on detection methods such as torque calculations or pressure sensors which require pinching to occur before being triggered. The issue with these systems is that injury to passengers is required before the detection system can react to the pinching event. This reactive approach presents several concerns for passenger safety, especially when considering children on school bus as they are more likely to be victims of pinching events. While other direct detection methods exist, they are impractical for large openings and are susceptible to varying environmental conditions. This paper proposes ProGuard, a novel, preemptive vision-based pinch detection method which uses state-of-the-art object detection algorithms to identify potential pinching events in bus entryways. ProGuard is based on YOLOv11 nano and is designed to run in real-time on low-power edge hardware. ProGuard is trained to identify people and backpacks with test results yielding 86% accuracy in backpack identification and 79% accuracy in people identification. When using an accelerated camera, ProGuard was able to achieve 24 frames per second and an inference time of 125ms. These results show that ProGuard offers an efficient alternative to current reactive pinch detection systems while operating on affordable and low-cost consumer hardware.
--

# Code Map
The code based is laid out as such:

## Python Scripts
- main.py: This is the base script that calls train.py, gpu_verfy.py, sonyimx500.py, and yolo.py. This script coordinates training, exporting, and predicting using a live webcam.
- capture_data.py: This script is used to capture images and video on a Rasbperry Pi camera. 
- model_testing.py: This script is used to calculate model metrics against the testing dataset.
- predict.py: This script is used to run predictions on videos or image directories.
- roboflow_model_upload.py: This script was used to upload train models to roboflow to speed up dataset annotionation.
- sonyimx500.py: This script is used to upload a model exported in the imx format to a Sony IMX500 camera.
- tune_yolo.py: This script is used to tune YOLO models.
- oi_download.py: This script is used to download a subset of the images from the Open Images dataset.
- 

## Folders
- datasets: this folder contains the active dataset that is being used to train the model. The active dataset resides in a subfolder called "bus-aps".
- conda_environment: This folder holds the conda config file which can be used to load the libraries for this repository.
- runs: This folder contains every version trained model and the export varients.

# How to use the Repository

### Model Training
To train a version of ProGuard, download the dataset from Roboflow and extract it into the folder "datasets/bus-aps". Then, run the `main.py` script. When prompted, enter `y` when asked if you want to train a new model. This will start the training process with the following parameters:
- Epochs: 500
- Image size: 640
- Batch size: (auto)
- GPU enabled: True
- Hue augmentation: 0.25
- Saturation augmentation: 0.25
- Exposure augmentation: 0.5
- Translation augmentaiton: 25%
- Scale augmentation: 30%
- Flip (L/R) augmentation: 20%
- IoU: 70%

### Model Exporting
To export the model to a different architecture (such as NCNN, ONNX, or IMX), use the `main.py` script. After selecting your trained model, enter `y` when asked if you want to export the model. Then, using the numbers displayed on screen, enter the model format you want to export to.

### Live Webcam Testing
To test the model with a live webcam, run the `main.py` script. After selecting your model and optionally exporting it, the script will automatically start running live detections using your webcam. Average inference time and FPS will be displayed once the script is stopped. To close the detection window, press the `q` button on your keyboard.


