from tqdm import tqdm
from skimage.feature import hog
import os
import cv2
import numpy as np
import pickle
from rembg import remove


real_image_folder = '../sketchydata/photo/test'
real_image_paths = []
sketch_image_folder = '../sketchydata/sketch/test/'
sketch_image_paths = []
for category in os.listdir(real_image_folder):
    real_image_category_path = os.path.join(real_image_folder, category)
    real_image_category_paths = [os.path.join(real_image_category_path, photo_image)
                                 for photo_image in os.listdir(real_image_category_path)]
    real_image_paths.extend(real_image_category_paths)

    sketch_image_category_path = os.path.join(sketch_image_folder, category)
    sketch_image_category_paths = [os.path.join(sketch_image_category_path, sketch_image)
                                   for sketch_image in os.listdir(sketch_image_category_path)]
    sketch_image_paths.extend(sketch_image_category_paths)

pixels_per_cell = (16, 16)

real_features = []
for real_image_path in tqdm(real_image_paths, desc="Extracting Real Features"):
    image = cv2.imread(real_image_path, cv2.IMREAD_GRAYSCALE)

    removebg_image = remove(image)
    removebg_image = cv2.cvtColor(removebg_image[:, :, :3], cv2.COLOR_BGR2GRAY)

    real_feature, _ = hog(image, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), visualize=True)
    real_features.append(real_feature)
real_features = np.array(real_features)
real_features = real_features.reshape(len(real_features), -1)

feature_folder = 'features_16x16_rmbg'
os.makedirs(feature_folder, exist_ok=True)
with open(f'{feature_folder}/real_features.pkl', 'wb') as file:
    pickle.dump(real_features, file)

sketch_features = []
for sketch_image_path in tqdm(sketch_image_paths, desc="Extracting Sketch Features"):
    image = cv2.imread(sketch_image_path, cv2.IMREAD_GRAYSCALE)

    sketch_feature, _ = hog(image, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), visualize=True)
    sketch_features.append(sketch_feature)
sketch_features = np.array(sketch_features)
sketch_features = sketch_features.reshape(len(sketch_features), -1)

with open(f'{feature_folder}/sketch_features.pkl', 'wb') as file:
    pickle.dump(sketch_features, file)