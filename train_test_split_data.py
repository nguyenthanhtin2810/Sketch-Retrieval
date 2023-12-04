import os
import shutil
from sklearn.model_selection import train_test_split

root = 'sketchydata'
# shutil.rmtree(root)
os.makedirs(root, exist_ok=True)

photo_folder = '256x256/photo/tx_000000000000'
sketch_folder = '256x256/sketch/tx_000000000000'

categories = os.listdir(photo_folder)
for category in categories:
    os.makedirs(f'{root}/photo/train/{category}', exist_ok=True)
    os.makedirs(f'{root}/photo/test/{category}', exist_ok=True)
    os.makedirs(f'{root}/sketch/train/{category}', exist_ok=True)
    os.makedirs(f'{root}/sketch/test/{category}', exist_ok=True)

    photo_images = os.listdir(f'{photo_folder}/{category}')

    train_photo_images, test_photo_images = train_test_split(photo_images, test_size=0.2, random_state=33)

    for file in train_photo_images:
        shutil.copy(f'{photo_folder}/{category}/{file}', f'{root}/photo/train/{category}/{file}')
    for file in test_photo_images:
        shutil.copy(f'{photo_folder}/{category}/{file}', f'{root}/photo/test/{category}/{file}')

    for sketch_image in os.listdir(f'{sketch_folder}/{category}'):
        main_image_name = sketch_image.split('-')[0] + '.jpg'
        if main_image_name in train_photo_images:
            shutil.copy(f'{sketch_folder}/{category}/{sketch_image}',
                        f'{root}/sketch/train/{category}/{sketch_image}')
        else:
            shutil.copy(f'{sketch_folder}/{category}/{sketch_image}',
                        f'{root}/sketch/test/{category}/{sketch_image}')

