# Importing the necessary packages

import argparse
from detector import MaskDetection
import cv2

# Constructing the argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True,
	help="path to input image")

args = vars(ap.parse_args())

image = cv2.imread(args['image'])

detector = MaskDetection()
detector.preprocess_image(image)

label, prob, bbox = detector.process_detections(image)


(startX, startY), (endX, endY) = bbox
color = (0, 255 ,0) if label else (0, 0, 255)
textLabel = "Mask Detected" if label else "No Mask"

cv2.putText(image, textLabel, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)