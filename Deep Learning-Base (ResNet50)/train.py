import torch.optim
from dataset import SketchDataset
from model import SketchResNet50
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    root = "../sketchydata"
    train_dataset = SketchDataset(root, train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    test_dataset = SketchDataset(root, train=False, transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=4,
    )

    if os.path.isdir("tensorboard"):
        shutil.rmtree("tensorboard")
    if not os.path.isdir("trained_models"):
        os.mkdir("trained_models")
    writer = SummaryWriter("tensorboard")
    model = SketchResNet50(num_classes=125).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    iters = len(train_dataloader)
    stop_flag = 0
    best_acc = 0
    for epoch in range(100):
        model.train()
        progess_bar = tqdm(train_dataloader, colour="green")
        for iter, (image_batch, label_batch) in enumerate(progess_bar):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            # forword
            outputs, _ = model(image_batch)
            loss_value = criterion(outputs, label_batch)
            progess_bar.set_description(f"Epoch [{epoch+1}/{100}]. Iteration [{iter+1}/{iters}]. Loss [{loss_value:.3f}]")
            writer.add_scalar("Train/Loss. ", loss_value, epoch*iters + iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (image_batch, label_batch) in enumerate(test_dataloader):
            all_labels.extend(label_batch)
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            with torch.no_grad():
                predictions, _ = model(image_batch)
                indices = torch.argmax(predictions, dim=1)
                all_predictions.extend(indices.cpu())
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"\nEpoch {epoch + 1}. Accuracy: {accuracy}\n")
        writer.add_scalar("Val/Accuracy. ", accuracy, epoch)

        if accuracy > best_acc:
            stop_flag = 0
            best_acc = accuracy
            checkpoint = {
                "best_acc": best_acc,
                "model": model.state_dict(),
            }
            torch.save(checkpoint, f"trained_models/best_model.pt")
        else:
            stop_flag += 1

        if stop_flag == 10:
            break


