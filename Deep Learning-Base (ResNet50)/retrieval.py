import cv2
import os
from model import SketchResNet50
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from PIL import Image

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

real_image_folder = f'../sketchydata/photo/test'
real_image_paths = []
for category in os.listdir(real_image_folder):
    real_image_category_path = os.path.join(real_image_folder, category)
    real_image_category_paths = [os.path.join(real_image_category_path, photo_image)
                                 for photo_image in os.listdir(real_image_category_path)]
    real_image_paths.extend(real_image_category_paths)

sketch_image_path = f'../sketchydata/sketch/test/geyser/n09288635_10240-1.png'

model = SketchResNet50(num_classes=125).to(device)
cp = torch.load("trained_models/model_lr0.001_epoch50.pt")
model.load_state_dict(cp)
model.eval()

saved_feature = 'features_lr0.001_epoch50'
real_features = torch.load(f'{saved_feature}/real_features.pt')

sketch_image = image_processer(sketch_image_path).to(device)
with torch.no_grad():
    _, sketch_feature = model(sketch_image)
    distances = 1 - torch.cosine_similarity(sketch_feature, torch.cat(real_features))
    indices = torch.argsort(distances)

print("Sketch image path:", sketch_image_path)
image = cv2.imread(sketch_image_path)
cv2.imshow("Sketch image", image)

for i, index in enumerate(indices[:5]):
    print("Real image path:", real_image_paths[index])
    image = cv2.imread(real_image_paths[index])
    cv2.imshow(f"Real image {i+1}", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
