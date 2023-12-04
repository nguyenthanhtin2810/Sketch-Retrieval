from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split

class SketchDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.root = root
        self.transform = transform

        photo_folder = os.path.join(root, "photo")
        sketch_folder = os.path.join(root, "sketch")
        if train:
            mode = "train"
        else:
            mode = "test"
        photo_mode = os.path.join(photo_folder, mode)
        sketch_mode = os.path.join(sketch_folder, mode)
        self.categories = os.listdir(photo_mode)

        for i, category in enumerate(self.categories):
            photo_category_path = os.path.join(photo_mode, category)
            sketch_category_path = os.path.join(sketch_mode, category)

            photo_image_paths = [os.path.join(photo_category_path, photo_image) for photo_image in os.listdir(photo_category_path)]
            sketch_image_paths = [os.path.join(sketch_category_path, sketch_image) for sketch_image in os.listdir(sketch_category_path)]
            image_category_paths = photo_image_paths + sketch_image_paths

            self.image_paths.extend(image_category_paths)
            self.labels.extend([i] * len(image_category_paths))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    transform = Compose([
        # Resize((224, 224)),
        # ToTensor(),
    ])
    dataset = SketchDataset(root="../sketchydata", train=True, transform=transform)
    image, label = dataset.__getitem__(2525)
    lenght = dataset.__len__()
    print(lenght)
    print(dataset.categories[label])
    image.show()

    # print(image, image.shape, image.dtype)


    # train_dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=4,
    #     shuffle=True
    # )
    # for images, labels in train_dataloader:
    #     print(images.shape)
    #     print(labels)