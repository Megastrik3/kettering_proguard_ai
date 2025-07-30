"""
This script captures images and videos using the Picamera2 library.
It allows the user to specify the number of images to capture and the duration of video recording.

A ribbon cable camera must be used. USB webcams are not supported.
"""

import os
from picamera2 import Picamera2, Preview
import datetime
from time import sleep

print("Starting capture...")
camera = Picamera2()
# Set camera capture size
camera_config = camera.create_still_configuration(main={"size": (640, 640)}, lores={"size": (640, 640)}, display="lores")
camera.configure(camera_config)
camera.start_preview(Preview.QTGL)
camera.start()
sleep(2)  # Allow time for the camera to adjust


while (True):
    num_cap = input("How many images do you want to capture: ")
    if num_cap.isdigit() and int(num_cap) > 0:
        num_cap = int(num_cap)
        break
    else:
        print("Please enter a valid positive integer.")
        continue

for i in range(num_cap):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{i+1}_{timestamp}.jpg"
    print(f"Capturing image {i+1} as {filename}...")
    camera.capture_file(f'./data-capture/{filename}')
    print(f"Image {i+1} captured successfully.")
    sleep(1)

print("All images captured successfully.")
camera.stop_preview()
camera.stop()
print("Camera stopped and resources released.")
sleep(2)

from picamera2.encoders import H264Encoder

# Set video capture size
video_config = camera.create_video_configuration(main={"size": (640, 640)}, lores={"size": (640, 640)}, display="lores")
camera.configure(video_config)
# Set encoder type and bitrate
encoder = H264Encoder(bitrate=10000000)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = "./data-capture/video_" + timestamp + ".mp4"

while (True):
    num_rec = input("How many seconds do you want to record (0 to cancel): ")
    if num_rec.isdigit() and int(num_rec) > 0:
        num_rec = int(num_rec)
        break
    elif num_rec == '0':
        print("Recording cancelled.")
        camera.stop_preview()
        camera.close()
        exit()
    else:
        print("Please enter a valid positive integer.")
        continue

camera.start_preview(Preview.QTGL)
camera.start_recording(encoder, output)

print(f"Recording video for {num_rec} seconds...")
sleep(num_rec)
camera.stop_recording()
print(f"Video saved as {output}.")
camera.stop_preview()
camera.close()
