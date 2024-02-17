import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from imgcluster.models import AutoEncoder
from imgcluster.dataset import ImageDataset
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Unsupervised Image Clustering Model Training")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("-o", "--output", default="./checkpoints", type=str)
    parser.add_argument(
        "-i", "--input", default="./dataset", type=str, help="path to images"
    )
    return parser.parse_args()


def train(args: dict):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    model = AutoEncoder()

    dataset = ImageDataset(args["input"], transforms=transform)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args["batch_size"], shuffle=True
    )

    device = args["device"]

    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = args["epochs"]

    model.to(device)

    for epoch in range(epochs):
        for x in tqdm(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = mse(output, x)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch [{epoch+1}/{epochs}]")

    torch.save(model.state_dict(), args["output"] + "/model.pt")

def make_dirs(args):
    Path(args["output"]).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    logging.info(args)
    make_dirs(args)
    train(vars(args))
