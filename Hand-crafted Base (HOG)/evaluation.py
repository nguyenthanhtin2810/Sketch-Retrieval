from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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

with open('features/real_features.pkl', 'rb') as file:
    real_features = pickle.load(file)

with open('features/sketch_features.pkl', 'rb') as file:
    sketch_features = pickle.load(file)

similarities_matrix = 1 - cosine_similarity(sketch_features, real_features)

top_1, top_5, top_10, k = 0, 0, 0, 10
progess_bar = tqdm(similarities_matrix, colour="green")
num_sketches = len(sketch_features)
for iter, similarities_row in enumerate(progess_bar):
    main_image_name = sketch_image_paths[iter].split("\\")[-1].split("-")[0]

    top_10_indices = np.argsort(similarities_row)[:k]
    top_10_similar_image = [real_image_paths[i].split("\\")[-1].split(".")[0] for i in top_10_indices]

    if main_image_name == top_10_similar_image[0]:
        top_1 += 1
    if main_image_name in top_10_similar_image[:5]:
        top_5 += 1
    if main_image_name in top_10_similar_image[:10]:
        top_10 += 1

    progess_bar.set_description(f"Top 1 [{(top_1/num_sketches):.3f}]. Top 5 [{(top_5/num_sketches):.3f}]. Top 10 [{(top_10/num_sketches):.3f}]")

print("Top 1 accuracy: ", top_1/num_sketches)
print("Top 5 accuracy: ", top_5/num_sketches)
print("Top 10 accuracy: ", top_10/num_sketches)

with open("eval.txt", "w") as file:
    file.write(f"Top 1 accuracy:  {top_1 / num_sketches}\n")
    file.write(f"Top 5 accuracy:  {top_5 / num_sketches}\n")
    file.write(f"Top 10 accuracy:  {top_10 / num_sketches}\n")