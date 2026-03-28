import os, shutil
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

model_path = input("Enter name of keras file (with extension): ")
#appends file extension if they forgot
if(model_path[-1:-6] != ".keras"):
    model_path += ".keras"
print("Successfully loaded: " + model_path)
model = tf.keras.models.load_model(model_path)
print("===============================")

