import argparse
import logging
import multiprocessing
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pinaka.model import AutoEncoder
from pinaka.dataset import ImageDataset
from pinaka.utils.io import load_json
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from pathlib import Path

logging.basicConfig(level=logging.INFO)
NUM_WORKER = multiprocessing.cpu_count() // 2


def parse_args():
    parser = argparse.ArgumentParser("PreTrian Task Model Training ")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--imgsz", default=32, type=int)
    parser.add_argument("--ckpt", default="./checkpoints", type=str)
    parser.add_argument("--data", default="./dataset", type=str)
    parser.add_argument("--test", default="./dataset", type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--config", default=None, type=str)
    return parser.parse_args()


def create_dataset(data, test, batch_size, imgsz):

    transform = transforms.Compose(
        [transforms.Resize((imgsz, imgsz)), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(data, transforms=transform)
    test_dataset = ImageFolder(test, transforms=transform)
    classes = train_dataset.classes

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader, classes

def train(model,train_dataloader,criterion,optimizer):


def pretrain(
    epochs=100,
    batch_size=64,
    device="cuda",
    data=None,
    test=None,
    imgsz=96,
    ckpt="./checkpoints/",
    weights=None,
    **kwargs,
):
    train_dataloader, test_dataloader, classes = create_dataset(data, test, batch_size, imgsz)

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1]=nn.Linear(in_features=1280,out_features=len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    best_weights = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    for epoch in range(epochs):
        running_corrects = 0
        running_total = 0
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(output, 1)
            corrects = torch.sum(labels == predictions)
            running_corrects += corrects.cpu().detach().item()
            running_total += torch.numel(labels)

        if loss < best_loss:
            best_loss = loss
            best_weights = copy.deepcopy(model.state_dict())

        for x in tqdm(test_dataloader):

            _, predictions = torch.max(output, 1)
            corrects = torch.sum(labels == predictions)
            running_corrects += corrects.cpu().detach().item()
            running_total += torch.numel(labels)

        if loss < best_loss:
            best_loss = loss
            best_weights = copy.deepcopy(model.state_dict())

        logging.info(f"Epoch: [{epoch+1}/{epochs}], Loss: {loss.mean()}")
        if epoch % 10 == 0 and epoch:
            torch.save(model.state_dict(), ckpt + f"/checkpoint-{epoch}.pt")
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), ckpt + "/best.pt")


def make_dirs(ckpt="./checkpoints/", **kwargs):
    Path(ckpt).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    if args['config']:
        args = load_json(args['config'])
    logging.info(args)
    make_dirs(**args)
    pretrain(**args)
