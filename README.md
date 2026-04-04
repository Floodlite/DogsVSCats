🐶 **Dogs VS Cats** 🐱

A deep classification network that can distinguish between real-life images of cats and dogs

Usage:
1. Download the DogsVSCats.py and model_.keras file
2. Run the program
3. Enter the path to the keras file
4. Enter the path to your own image file
5. See what the model thinks

You can also train the model yourself to generate new keras files:
1. Create two folders in your directly called "Dogs" and "Cats"
2. Set load_file variable to false
3. Optionally set the display_graphs to true to view the loss reports over time
4. Run the DogsVSCats.py program
5. Enter the number of epochs to train for
6. View your files in the explorer for the newly produced model.keras checkpoint
7. Use your files wherever you want

In addition, ImagePreprocessing.py can be used to generate augmented images with the following filterts:
* RGB channels
* Rotation
* Translation
* Blue
* Threshold
* Contrast
* Brightness
* Add/reduce noise

Training images sourced from Kaggle:[ https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset](url)
