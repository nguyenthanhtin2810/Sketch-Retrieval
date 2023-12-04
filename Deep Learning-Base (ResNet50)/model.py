import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SketchResNet50(nn.Module):
    def __init__(self, num_classes=125):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.backbone.fc
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.fc(features)

        return x, features

if __name__ == '__main__':
    model = SketchResNet50()
    input_data = torch.rand(1, 3, 224, 224)
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()
    while True:
        x, features = model(input_data)
        print(x.shape, features.shape)
        break
