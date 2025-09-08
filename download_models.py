import os
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model

# Create directories if they don't exist
os.makedirs("face_detector", exist_ok=True)

# Download face detector model
print("[INFO] Downloading face detector model...")
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
weights_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

urllib.request.urlretrieve(prototxt_url, "face_detector/deploy.prototxt")
urllib.request.urlretrieve(weights_url, "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

# Create a simple mask detector model based on MobileNetV2
print("[INFO] Creating mask detector model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Save the model
model.save("mask_detector.model")
print("[INFO] Mask detector model created and saved.")