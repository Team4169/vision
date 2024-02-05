import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt

def tflite_detect_webcam(modelpath, lblpath, min_conf=0.5):

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
        print(labels)

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and process frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
        classes = interpreter.get_tensor(output_details[3]['index'])[0] 
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        imH, imW, _ = frame.shape

		# Draw detection results on the frame
        for i in range(len(scores)):
              if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
				# Get bounding box coordinates and draw box
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

				# Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Display the resulting frame
        cv2.imshow('Object detector', frame)
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture when everything is done
    cap.release()
    cv2.destroyAllWindows()

# Define the paths to your model and label file
model_path = 'detect.tflite'
label_path = 'labelmap.pbtxt'

# Set the minimum confidence threshold for detection
min_conf_threshold = 0.5

# Call the function to start object detection with webcam
tflite_detect_webcam(model_path, label_path, min_conf_threshold)