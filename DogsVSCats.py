import os, shutil, pathlib
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, losses
import random
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def image_formatting(image, label):
  image = tf.cast(image, tf.float32)
  image = applications.resnet_v2.preprocess_input(image)
  image = tf.image.resize(image, (180, 180))
  return image, label


def make_subset(subset_name, start_index, end_index):
    for category in ("Dogs", "Cats"):
        #original_dir = pathlib.Path("\\" + category)
        dir = subset_name + "\\" + category
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass
        fnames = [f"{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            try:
                print(fname)
                shutil.copyfile(src=category + "\\" + fname,
                                dst=dir + "\\" + fname)
            except FileNotFoundError:
                print("Not found: " + fname)
                pass

make_subset("Train", start_index=0, end_index=1000)
make_subset("Val", start_index=1000, end_index=1500)
make_subset("Test", start_index=1500, end_index=2500)

train_dataset = image_dataset_from_directory(
    "/" + "Train",
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    "/" + "Val",
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    "/" + "Test",
    image_size=(180, 180),
    batch_size=32)

#add in pre-processing effects
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

"""
model = models.Sequential()
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
"""
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.layers[0].trainable = False
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


