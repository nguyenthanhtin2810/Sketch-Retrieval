from tqdm import tqdm
from skimage.feature import hog
import os
import cv2
import numpy as np
import pickle

os.makedirs("features", exist_ok=True)

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

real_features = []
for real_image_path in tqdm(real_image_paths, desc="Extracting Real Features"):
    image = cv2.imread(real_image_path, cv2.IMREAD_GRAYSCALE)

    _, real_feature = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    real_features.append(real_feature)
real_features = np.array(real_features)
real_features = real_features.reshape(len(real_features), -1)

with open('features/real_features.pkl', 'wb') as file:
    pickle.dump(real_features, file)

sketch_features = []
for sketch_image_path in tqdm(sketch_image_paths, desc="Extracting Sketch Features"):
    image = cv2.imread(sketch_image_path, cv2.IMREAD_GRAYSCALE)

    _, sketch_feature = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    sketch_features.append(sketch_feature)
sketch_features = np.array(sketch_features)
sketch_features = sketch_features.reshape(len(sketch_features), -1)

with open('features/sketch_features.pkl', 'wb') as file:
    pickle.dump(sketch_features, file)