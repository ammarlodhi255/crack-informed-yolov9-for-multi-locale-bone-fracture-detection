
import warnings
warnings.filterwarnings("ignore")
from keras_preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import os
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image



datagen = ImageDataGenerator(
        width_shift_range = 0.2,
        height_shift_range = 0.1,
        horizontal_flip = True,
        shear_range = 0.2,
        brightness_range = (0.7, 1.3),
        zoom_range = 0.1,
        zca_whitening = True,
        fill_mode='constant'
)



INPUT_DIR = "/cluster/home/ammaa/Downloads/FracAtlas/classes images/Fractured"
OUTPUT_DIR = "/cluster/home/ammaa/Downloads/FracAtlas/classes images/Fractured-Aug"

current_transformations = 5
max_imgs = len(os.listdir(INPUT_DIR)) * current_transformations
img_count = 0
    
for file in os.listdir(INPUT_DIR):
    img = Image.open(os.path.join(INPUT_DIR, file))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    images = []
    for batch in datagen.flow(x, batch_size=1):
        images.append(image.array_to_img(batch[0]))
        i += 1
        if i == current_transformations:
            break
    
    for i in range(0, current_transformations):
        images[i].save(f"{OUTPUT_DIR}/{file[:-4]}-{i}.jpg")
        
    img_count += current_transformations
    if img_count > (max_imgs): break

