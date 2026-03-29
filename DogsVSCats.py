import os, shutil
import PIL
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array
import matplotlib.pyplot as plt
import time
import numpy as np
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def make_subset(subset_name, start_index, end_index):
    for category in ("Dogs", "Cats"):
        target_directory = subset_name + "\\" + category
        #target_directory = "Temp"
        try:
            os.makedirs(target_directory + "\\" + category)
        except FileExistsError:
            pass
        file_names = [os.path.join(path, name) for path, subdir, files
                     in os.walk(category) for name in files if name.endswith(".png") or name.endswith("jpg")]
        print(file_names)
        for file_name in file_names:
            print(file_name)
            #print(type(file_name))
            try:
                shutil.copyfile(src= file_name,
                                dst=target_directory + "\\" + file_name)
            except FileNotFoundError:
                print("Not found: " + file_name)
                #continue


recreate_subsets = False
if(recreate_subsets):
    print("[Creating subsets, please hold]")
    time.sleep(4.5)
    print("==========================")

    make_subset("Train", start_index=0, end_index=1000)
    make_subset("Val", start_index=1000, end_index=1500)
    make_subset("Test", start_index=1500, end_index=2500)

    train_dataset = image_dataset_from_directory(
        "Train",
        image_size=(180, 180),
        batch_size=32,
        labels="inferred",
        label_mode="binary",
        shuffle=True)
    validation_dataset = image_dataset_from_directory(
        "Val",
        image_size=(180, 180),
        batch_size=32,
        labels="inferred",
        label_mode="binary",
        shuffle=True)
    test_dataset = image_dataset_from_directory(
        "Test",
        image_size=(180, 180),
        batch_size=32,
        labels="inferred",
        label_mode="binary",
        shuffle=True)

    #add in custom pre-processing effects later
    data_augmentation = keras.Sequential(
        [layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        ])

print("==========================")
#determines whether or not a saved model will be used
load_file = True

if(load_file):
    model_path = input("Enter name of keras file (with extension): ")
    print("Successfully loaded: " + model_path)
    model = tf.keras.models.load_model(model_path)
    print("==========================")
else:
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0/255)(x)
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

    print("==========================")
    num_epochs = int(input("Number of epochs: "))
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=validation_dataset)
    #number designates the number of epochs that transpired
    model.save("model_" + str(num_epochs) + ".keras")
    print("Saved: model_" + str(num_epochs) + ".keras")
    print("==========================")

print(model.summary())

display_graphs = False
if(display_graphs and not load_file):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and Validation Accuracy (" + str(num_epochs) + " epochs)")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

while(True):
    print("===============================================")
    image_name = input("Enter name of image (with extension): ")
    if(image_name == "Goodbye"):
        message = random.randint(0, 1)
        if(message == 0):
            print("Woof! Have a good night!")
        else:
            print("Meow! Have a good night!")

    try:
        #height x width
        loaded_image = load_img(image_name, target_size=(180, 180, 3))
    except FileNotFoundError:
        print("Image not found")
        print("Check your spelling or file extension")
        continue
    except PIL.UnidentifiedImageError:
        print("Bad image format used")
        print("Please refrain from using any formats besides the following:")
        print(" * .jpg")
        print(" * jpeg")
        print(" * jpeg")
        print(" * png")
        print(" * bmp")
        print(" * gif")

    image_array = img_to_array(loaded_image)
    #expand to three dimensions
    expanded_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(expanded_array)
    predicted_class = predictions > 0.5
    predicted_class = predicted_class.astype(int)

    if(predicted_class == 0):
        print_predicted_label = "Cat"
    else:
        print_predicted_label = "Dog"

    plt.imshow(loaded_image)
    plt.title(f"Predicted Label: {print_predicted_label}")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(20)
    plt.close()

    if(predicted_class == 0):
        print(image_name + " predicted as cat!")
    else:
        print(image_name + " predicted as dog!")