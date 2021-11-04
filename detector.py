import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


class MaskDetection:
    def __init__(self):
        self.protoTxtPath = os.path.sep.join(["face_detector","deploy.prototxt"])
        self.weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])

        self.net = cv2.dnn.readNet(self.protoTxtPath, self.weightsPath)
        self.model = load_model("mask_detector.model")

        self.detections = None
    
    def preprocess_image(self, image):
        # Construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the face detections
        print("[INFO] Computing face detections...")
        
        self.net.setInput(blob)
        self.detections = self.net.forward()

    def process_detections(self, image):
        h,w = image.shape[:2]

        # Loop over the detections
        for i in range(0, self.detections.shape[2]):
            # Extract the confidence (i.e. probability) associated with
            # the detection
            confidence = self.detections[0, 0, i, 2]

            # Filter out weak detections by ensuring confidence is 
            # greater than the minimum confidence
            
            if confidence > 0.5:
                # Compute the (x, y)-coordinates of the bounding box for
                # the object
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w-1, endX), min(h-1, endY))

                # Extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis = 0)

                # Pass the face through the model to determine if the face
                # has a mask or not
                (mask, withoutMask) = self.model.predict(face)[0]

                # Determine the class label and color we'll use to draw 
                # the bounding box and text
                label =  mask > withoutMask

                # Include the probability in the label
                pred_prob = max(mask, withoutMask)

                # Return the predicted label and the bounding box co-ordinates
                return label, pred_prob, ((startX, startY), (endX, endY))