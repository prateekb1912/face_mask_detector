# Import necessary packages

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from imutils import paths

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from keras.layers import AveragePooling2D, Dropout, Flatten, Conv2D
from keras.layers import Dense, Input, Activation
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow.compat.v1 as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

session = tf.InteractiveSession(config = config)

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# Construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Path to output face detector model")

args = vars(ap.parse_args())

# Initialize the learning rate, number of epochs, and the batch size
INIT_LR = 1e-4
EPOCHS = 8
BATCH_SIZE = 8

# Grab the list of images in our dataset directory, then initialize the list of data and class labels
print("[INFO] Loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Loop over the image paths
for imgPath in imagePaths:
    # Extract the class label from the filename
    label = imgPath.split(os.path.sep)[-2]

    # Load the input image (224x224) and preprocess it
    image = load_img(imgPath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # Update the data and class labels  
    data.append(image)
    labels.append(label)

# Convert the data and labels into NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Performing one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Splitting the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state = 10)

print(trainX.shape)
print(testX.shape)


# Construct the training image generator for data augmentation
# aug = ImageDataGenerator(
#     rotation_range=20,
#     zoom_range=0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest"
# )

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process

for layer in baseModel.layers:
	layer.trainable = False


model.summary()

# Compile the model
print("[INFO] Compiling Model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics = ["accuracy"])

# Train the head of the network
print("[INFO] Training Model...")
H = model.fit(
    x = trainX,
    y = trainY,
    batch_size = BATCH_SIZE,
    validation_data = (testX, testY),
    epochs = EPOCHS
)

# Make predictions on the testing set
print("[INFO] Making predictions...")

predIdxs = model.predict(testX, batch_size = BATCH_SIZE)

predIdxs = np.argmax(predIdxs, axis=1)

# Serialize the model to disk
print("[INFO] Saving mask detector model...")
model.save(args["model"], save_format="h5")


# Show a nicely formatted classification report
print(classification_report(testY.argmax(axis = 1), predIdxs, target_names=lb.classes_))