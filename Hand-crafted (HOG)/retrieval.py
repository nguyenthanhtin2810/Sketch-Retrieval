import cv2
import numpy as np
from skimage.feature import hog
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

real_image_folder = '../sketchydata/photo/test'
real_image_paths = []
for category in os.listdir(real_image_folder):
    real_image_category_path = os.path.join(real_image_folder, category)
    real_image_category_paths = [os.path.join(real_image_category_path, photo_image)
                                 for photo_image in os.listdir(real_image_category_path)]
    real_image_paths.extend(real_image_category_paths)

sketch_image_path = '../sketchydata/sketch/test/bear/n02131653_851-6.png'

saved_feature = 'features_16x16'
with open(f'{saved_feature}/real_features.pkl', 'rb') as file:
    real_features = pickle.load(file)

sketch_image = cv2.imread(sketch_image_path, cv2.IMREAD_GRAYSCALE)
sketch_feature, _ = hog(sketch_image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
sketch_feature = sketch_feature.reshape(1, -1)

similarities = 1 - cosine_similarity(sketch_feature, real_features)
indices = np.argsort(similarities)[0]

print("Sketch image path:", sketch_image_path)
image = cv2.imread(sketch_image_path)
cv2.imshow("Sketch image", image)

for i, index in enumerate(indices[:5]):
    print("Real image path:", real_image_paths[index])
    image = cv2.imread(real_image_paths[index])
    cv2.imshow(f"Real image {i+1}", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
