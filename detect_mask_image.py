# Importing the necessary packages

import argparse
from detector import MaskDetection
import cv2

# Constructing the argument parser
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")

args = vars(ap.parse_args())
                

# cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
# cv2.rectangle(image, , color, 2)

# Load the input image from disk, clone it, and grab the image
# spatial dimensions
image = cv2.imread('images_used/Friendsphoebe.jpg')

detector = MaskDetection()
detector.preprocess_image(image)
print(detector.process_detections(image))