import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import os
import skimage.data
import scipy.signal
from skimage import util, filters, morphology, color, transform
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.transform import SimilarityTransform
import skimage.exposure as sk_exposure

def save_image(mode, name, image, output_dir):
    plt.imshow(image, cmap="grey")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_dir + "/" + data_dir + "_" + str(index) + "_" + mode + ".jpg", bbox_inches='tight')

def save_image2(mode, name, image, output_dir):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_dir + "/" + data_dir + "_" + str(index) + "_" + mode + ".jpg", bbox_inches='tight')
    plt.close()

def blur(image, filter_size, name):
  kx = filter_size
  ky = filter_size
  g_uniform = np.ones([kx, ky]) / (kx + ky)
  img_uniform_convolution = scipy.signal.correlate2d(image, g_uniform, mode="valid")
  save_image(("blur"+ str(filter_size)), name, img_uniform_convolution, output_dir)

def red_channel(image, name):
  img_red = image.copy()
  img_red[:, :, 1] = np.zeros(image.shape[:2])
  img_red[:, :, 2] = np.zeros(image.shape[:2])
  save_image("red", name, img_red, output_dir)

def green_channel(image, name):
  img_green = image.copy()
  img_green[:, :, 0] = np.zeros(image.shape[:2])
  img_green[:, :, 2] = np.zeros(image.shape[:2])
  save_image("green", name, img_green, output_dir)

def blue_channel(image, name):
  img_blue = image.copy()
  img_blue[:, :, 0] = np.zeros(image.shape[:2])
  img_blue[:, :, 1] = np.zeros(image.shape[:2])
  save_image("blue", name, img_blue, output_dir)

def multi_channel(image, name):
  img_multi = 0.22989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
  save_image2("multicolor", name, img_multi, output_dir)

def exposure(image, expose, name):
  img3 = skimage.util.img_as_float(image)
  exposure_img = skimage.exposure.adjust_gamma(img3, expose)
  save_image(("exposure" + str(expose)), name, exposure_img, output_dir)

def contrast(image, name):
  p2, p98 = np.percentile(image, (2, 98))
  contrast_img = sk_exposure.rescale_intensity(image, in_range=(p2, p98))
  save_image("contrast", name, contrast_img, output_dir)

def brightness(image, exposure, name):
  img3 = skimage.util.img_as_float(image)
  bright_img = skimage.exposure.adjust_gamma(img3, exposure)
  save_image(("brightness" + str(exposure)), name, bright_img, output_dir)

def inject_noise(image, intensity, name):
  noise_intensity = intensity / 100
  loud_img = util.random_noise(image, mode='s&p', amount=noise_intensity)
  save_image(("inject_noise" + str(intensity)), name, loud_img, output_dir)

def reduce_noise(image, disk_radius, name):
  try:
    bw_img = color.rgb2gray(image)
    bw_img_uint8 = util.img_as_ubyte(bw_img)
    footprint = morphology.disk(disk_radius)
    silent_img = filters.rank.median(bw_img_uint8, footprint)
    save_image(("reduce_noise" + str(disk_radius)), name, silent_img, output_dir)
  except ValueError:
      pass

def threshold(image, name):
  thresh_value = filters.threshold_otsu(image)
  binary_img = image > thresh_value
  save_image("threshold", name, binary_img, output_dir)

def rotate(image, degree, name):
  rotated_image = skimage.transform.rotate(image, degree)
  save_image(("rotate" + str(degree)), name, rotated_image, output_dir)

def shift_x(image, x_shift, name):
  tform = SimilarityTransform(translation=(x_shift, 0))
  warped = warp(image, tform)
  save_image(("x-shift" + str(x_shift)), name, warped, output_dir)

def shift_y(image, y_shift, name):
  tform = SimilarityTransform(translation=(0, y_shift))
  warped = warp(image, tform)
  save_image(("y-shift" + str(y_shift)), name, warped, output_dir)





data_dir = input("Enter name of source folder: ")
output_dir = input("Enter name of output folder: ")
all_files = [os.path.join(path, name) for path, subdir, files
    in os.walk(data_dir) for name in files if name.endswith(".png") or name.endswith("jpg")]

full_size = False
for index, file in enumerate(all_files):
    print("Current: " + file + " (" + str(round(round((all_files.index(file)+1)/len(all_files),4)*100,4)) + "% done)")
    img = Image.open(file)
    greyscale_img = img.convert('L')
    matrix_img = np.asarray(greyscale_img)
    matrix_img_float = util.img_as_float(matrix_img)
    img_clahe = sk_exposure.equalize_adapthist(matrix_img_float, clip_limit=0.03)
    matrix_img2 = np.asarray(img)
    plt.imshow(img)
    plt.close()

    if(full_size):
        for filter_size in range(5, 20, 5):
            blur(matrix_img, filter_size, file)
        for expose in range(3, 6):
            exposure(matrix_img2, expose, file)
        for brightness_val in range(4, 8, 2):
            bright = brightness_val / 10
            brightness(matrix_img2, bright, file)
        for noise in range(5, 15, 5):
            inject_noise(matrix_img2, noise, file)
        for denoise in range(6, 10, 2):
            reduce_noise(matrix_img2, denoise, file)
        for degree in range(45, 315, 45):
            rotate(matrix_img2, degree, file)
        for x in range(-100, 100, 25):
            shift_x(matrix_img2, x, file)
        for y in range(-100, 100, 25):
            shift_y(matrix_img2, y, file)
    else:
        blur(matrix_img, 7, file)
        exposure(matrix_img2, 4, file)
        brightness(matrix_img2, 0.4, file)
        inject_noise(matrix_img2, 8, file)
        reduce_noise(matrix_img2, 5, file)
        rotate(matrix_img2, 45, file)
        rotate(matrix_img2, -45, file)
        shift_x(matrix_img2, 75, file)
        shift_y(matrix_img2, 75, file)
        shift_x(matrix_img2, -75, file)
        shift_y(matrix_img2, -75, file)

    red_channel(matrix_img2, file)
    green_channel(matrix_img2, file)
    blue_channel(matrix_img2, file)
    multi_channel(matrix_img2, file)
    contrast(matrix_img2, file)
    threshold(matrix_img, file)