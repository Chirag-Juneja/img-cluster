import copy
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from .datasets.raw_dataset import RawDataset
from .models.effb0ae import EffB0AE
from torch.utils.data import DataLoader
from pinaka import NUM_WORKER
from tqdm import tqdm
from .utils.io import make_dirs


class Classificaiton:
    def __init__(self, model=None, device="cuda"):
        path = Path(model)
        self.model = EffB0AE()

        if path.is_file():
            try:
                model.load_state_dict(torch.load(path))
                logging.info(f"Model weights restored from {path}")
            except Exception as e:
                logging.warning(f"Weights load failed. {path}\n {e}")

        self.imgsz = 64
        self.device = device

    def _load_pretrain_data(self, imgsz, batch):
        if imgsz:
            self.imgsz = imgsz

        self.transform = transforms.Compose(
            [transforms.Resize((self.imgsz, self.imgsz)), transforms.ToTensor()]
        )

        dataset = RawDataset(data,self.transform)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=NUM_WORKER,
        )
        return dataloader

    def pretrain(self, data, epochs=100, batch=1024, ckpt="./checkpoints", imgsz=None):

        dataloader = self._load_pretrain_data(imgsz, batch)

        mse = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.to(self.device)

        make_dirs(ckpt)

        best_weights = copy.deepcopy(self.model.state_dict())

        best_loss = float("inf")

        for epoch in range(epochs):
            for x in tqdm(dataloader):
                x = x.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = mse(output, x)
                loss.backward()
                optimizer.step()
                if loss < best_loss:
                    best_loss = loss
                    best_weights = copy.deepcopy(self.model.state_dict())

            logging.info(f"Epoch: [{epoch+1}/{epochs}], Loss: {loss.mean()}")

            if epoch % 10 == 0 and epoch:
                torch.save(model.state_dict(), ckpt + f"/checkpoint-{epoch}.pt")

        self.model.load_state_dict(best_weights)

        torch.save(self.model.state_dict(), ckpt + "/best.pt")
        torch.save(self.model.encoder.state_dict(), ckpt + "/encoder.pt")

    def _load_classification_data(self, data, test, batch, imgsz):

        train_dataset = ImageFolder(data, transforms=self.transform)
        test_dataset = ImageFolder(test, transforms=self.transform)
        classes = train_dataset.classes

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=NUM_WORKER,
            pin_memory=True,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=NUM_WORKER,
            pin_memory=True,
        )
        return train_dataloader, test_dataloader, classes

    def train(self, data, test, epochs=100, batch=1024, ckpt="./checkpoints", imgsz=None):

        train_dataloader, test_dataloader, classes = _load_classification_data(data,test,batch,imgsz)
        mse = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.to(self.device)

        make_dirs(ckpt)

        best_weights = copy.deepcopy(self.model.state_dict())

        best_loss = float("inf")

        for epoch in range(epochs):
            for x in tqdm(dataloader):
                x = x.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = mse(output, x)
                loss.backward()
                optimizer.step()
                if loss < best_loss:
                    best_loss = loss
                    best_weights = copy.deepcopy(self.model.state_dict())

            logging.info(f"Epoch: [{epoch+1}/{epochs}], Loss: {loss.mean()}")

            if epoch % 10 == 0 and epoch:
                torch.save(model.state_dict(), ckpt + f"/checkpoint-{epoch}.pt")

        self.model.load_state_dict(best_weights)

        torch.save(self.model.state_dict(), ckpt + "/best.pt")
        torch.save(self.model.encoder.state_dict(), ckpt + "/encoder.pt")


