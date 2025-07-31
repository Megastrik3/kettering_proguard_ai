"""
This script performs real-time object detection using the YOLO model.
It captures video from the webcam, processes each frame, and displays detected objects with bounding boxes and
labels.
A ROI can also be specified for detecting only specific regions.
"""
import math
import time
from ultralytics import YOLO
# https://www.reddit.com/r/learnpython/comments/zxxsal/open_cv_video_from_webcam_takes_abnormally_long/
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

CAM_RESOLUTION = (640, 640)  # Set the camera resolution to 1920x1080

def main(model_file='yolo11n.pt'):
    print("Starting YOLO Object Detection...\n\n")
    cap = cv2.VideoCapture(0)
    # Set frame time baseline
    old_frame_time = 0
    new_frame_time = 0
    fps_sum = 0
    fps_counter = 0

    # Set camera resolution
    cap.set(3, CAM_RESOLUTION[0])
    cap.set(4, CAM_RESOLUTION[1])

    # Set the ROI location (X, Y) and size (W, H)
    roi_x, roi_y, roi_w, roi_h = 0, 0, 640, 640

    model = YOLO(model_file, task="detect")  # Load the custom model
    print("Model loaded successfully.")

    # Load class names from the model
    classNames = model.names
    print("Class names loaded successfully.")
    print(classNames)
    inference_times = []
    frame_count = 0

    while True:
        print("Capturing frame...")
        success, img = cap.read()
        frame_count += 1
        if not success:
            print("Failed to capture frames. Exiting...")
            break
        start_time = time.time()
        roi_frame = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        # Load the model and perform inference only showing objects with a confidence greater than 65%
        results = model(roi_frame, stream=True, conf=0.65)


        # Calculate FPS 
        # https://www.geeksforgeeks.org/python/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - old_frame_time)
        old_frame_time = new_frame_time
        fps = int(fps)
        fps_sum += fps
        fps = str(fps)
        fps_counter += 1
        confidence = 0


        # Draw and label boxes on detected objects
        # https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1 + roi_x), int(y1 + roi_y), int(x2 + roi_x), int(y2 + roi_y) # convert to int values


                cv2.rectangle(img, (x1, y1), (x2, y2), (10, 241, 2), 3)
            
                cls = int(box.cls[0])
                if int(box.cls[0]) > 0:
                    print("Class name -->", classNames[cls])
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->",confidence)
                

                
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 8)
                thickness = 2

                cv2.putText(img, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)

                
        cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)  # Draw ROI
        cv2.putText(img, fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (241, 2, 2), 2) # Display FPS

        # Create Window
        cv2.imshow(f'BUS-APS: {model_file}', img)

        end_time = time.time()
        inference_time = end_time - start_time
        if frame_count > 150:
            inference_times.append(inference_time)
        print(f"Inference time for frame: {inference_time:.4f} seconds")
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Average FPS:", fps_sum / fps_counter if fps_counter > 0 else 0)
    print("Average Inference Time:", sum(inference_times) / len(inference_times) if inference_times else 0)


if __name__ == '__main__':
    main()