import os
from model import SketchResNet50
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from PIL import Image
from tqdm import tqdm

os.makedirs("features", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((224, 224)),
    ToTensor()
])
def image_processer(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

root = '../sketchydata'
real_image_folder = f'{root}/photo/test'
real_image_paths = []
sketch_image_folder = f'{root}/sketch/test/'
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

model = SketchResNet50(num_classes=125).to(device)
# cp = torch.load("trained_models/best_model.pt")
# model.load_state_dict(cp["model"])

model.eval()

real_features = []
for real_image_path in tqdm(real_image_paths, desc="Extracting Real Features"):
    real_image = image_processer(real_image_path).to(device)
    with torch.no_grad():
        _, real_feature = model(real_image)
    real_features.append(real_feature)

torch.save(real_features, "features/real_features.pt")

sketch_features = []
for sketch_image_path in tqdm(sketch_image_paths, desc="Extracting Sketch Features"):
    sketch_image = image_processer(sketch_image_path).to(device)
    with torch.no_grad():
        _, sketch_feature = model(sketch_image)
    sketch_features.append(sketch_feature)

torch.save(sketch_features, "features/sketch_features.pt")

