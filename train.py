import argparse
import logging
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pinaka.model import AutoEncoder
from pinaka.dataset import ImageDataset
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path

logging.basicConfig(level=logging.INFO)
NUM_WORKER = multiprocessing.cpu_count() // 2


def parse_args():
    parser = argparse.ArgumentParser("Unsupervised Image Clustering Model Training ")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--imgsz", default=32, type=int)
    parser.add_argument("-ckpt", "--checkpoints", default="./checkpoints", type=str)
    parser.add_argument("--data", default="./dataset", type=str)
    return parser.parse_args()


def train(epochs, batch_size, device, checkpoints, data, imgsz):
    transform = transforms.Compose(
        [transforms.Resize((imgsz, imgsz)), transforms.ToTensor()]
    )

    model = AutoEncoder()

    dataset = ImageDataset(data, transforms=transform)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKER
    )

    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(epochs):
        for x in tqdm(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = mse(output, x)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch: [{epoch+1}/{epochs}], Loss: {loss.mean()}")
        if epoch % 10 == 0 and epoch:
            torch.save(model.state_dict(), checkpoints + f"/checkpoint-{epoch}.pt")

    torch.save(model.state_dict(), checkpoints + "/model.pt")


def make_dirs(args):
    Path(args["checkpoints"]).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    logging.info(args)
    make_dirs(args)
    train(**args)
