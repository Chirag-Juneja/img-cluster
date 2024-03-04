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
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--config", default=None, type=str)
    return parser.parse_args()


def pretrain(
    epochs=100,
    batch_size=64,
    device="cuda",
    data="./dataset",
    imgsz=96,
    ckpt="./checkpoints/",
    weights=None,
    **kwargs,
):
    transform = transforms.Compose(
        [transforms.Resize((imgsz, imgsz)), transforms.ToTensor()]
    )

    model = AutoEncoder()
    if weights:
        try:
            model.load_state_dict(torch.load(weights))
            logging.info(f"Model weights restored from {weights}")
        except Exception as e:
            logging.warning(f"Weights load failed. {weights}\n {e}")

    dataset = ImageDataset(data, transforms=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True,
    )

    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    best_weights = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    for epoch in range(epochs):
        for x in tqdm(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = mse(output, x)
            loss.backward()
            optimizer.step()
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
