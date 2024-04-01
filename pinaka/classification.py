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
from .models.effb0cl import EffB0CL
from torch.utils.data import DataLoader
from pinaka import NUM_WORKER
from tqdm import tqdm
from .utils.io import make_dirs
from collections import OrderedDict


class Classificaiton:
    def __init__(
        self, effb0ae=None, effb0cl=None, device="cuda", classes=None, save_period=10
    ):
        self.classes = classes
        self.effb0ae = EffB0AE()
        self.effb0cl = EffB0CL(classes=classes)

        if effb0cl:
            effb0cl_path = Path(effb0cl)
            self.load_effb0cl(effb0cl_path)

        if effb0ae:
            effb0ae_path = Path(effb0ae)
            self.load_effb0ae(effb0ae_path)

        self.imgsz = 64
        self.device = device

        self.save_period = save_period

    def load_effb0ae(self, effb0ae_path):
        if effb0ae_path.is_file():
            try:
                self.effb0ae.load_state_dict(torch.load(effb0ae_path))
                logging.info(f"Model weights restored from {effb0ae_path}")
            except Exception as e:
                logging.warning(f"Weights load failed. {effb0ae_path}\n {e}")

    def load_effb0cl(self, effb0cl_path):
        if effb0cl_path.is_file():
            try:
                self.effb0cl.load_state_dict(torch.load(effb0cl_path))
                logging.info(f"Model weights restored from {effb0cl_path}")
            except Exception as e:
                logging.warning(f"Weights load failed. {effb0cl_path}\n {e}")

    def _load_pretrain_data(self, data, imgsz, batch):
        self.transform = transforms.Compose(
            [transforms.Resize((self.imgsz, self.imgsz)), transforms.ToTensor()]
        )

        dataset = RawDataset(data, self.transform)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=NUM_WORKER,
        )
        return dataloader

    def pretrain(self, data, epochs=100, batch=1024, ckpt="./checkpoints", imgsz=None):
        if imgsz:
            self.imgsz = imgsz

        self.model = self.effb0ae

        dataloader = self._load_pretrain_data(data, imgsz, batch)

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

            if epoch % self.save_period == 0 and epoch:
                torch.save(model.state_dict(), ckpt + f"/checkpoint-{epoch}.pt")

        self.model.load_state_dict(best_weights)

        torch.save(self.model.state_dict(), ckpt + "/best-ae.pt")
        torch.save(self.model.encoder.state_dict(), ckpt + "/encoder.pt")

        self.effb0cl.features = self.model.encoder

    def _load_classification_data(self, data, test, batch, imgsz):
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.3),
                transforms.ToTensor(),
            ]
        )

        dataset, dataloader = {}, {}

        dataset["train"] = ImageFolder(data, transform=self.transform)
        dataset["test"] = ImageFolder(test, transform=self.transform)

        classes = dataset["train"].classes

        dataloader["train"] = DataLoader(
            dataset=dataset["train"],
            batch_size=batch,
            shuffle=True,
            num_workers=NUM_WORKER,
            pin_memory=True,
        )

        dataloader["test"] = DataLoader(
            dataset=dataset["test"],
            batch_size=batch,
            shuffle=True,
            num_workers=NUM_WORKER,
            pin_memory=True,
        )
        return dataset, dataloader, classes

    def load_features(self, features):
        if not isinstance(self.model, EffB0CL):
            raise Exception("Switch to classification model")

        if isinstance(features, str):
            self.model.features.load_state_dict(torch.load(features))

        if isinstance(features, OrderedDict):
            self.model.features = features

    def train(
        self,
        data,
        test,
        epochs=100,
        batch=1024,
        ckpt="./checkpoints",
        imgsz=None,
        features=None,
    ):
        if not self.classes:
            raise Exception("Classes not defined...")

        if imgsz:
            self.imgsz = imgsz

        self.model = self.effb0cl

        self.load_features(features)

        dataset, dataloader, classes = self._load_classification_data(
            data, test, batch, imgsz
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.to(self.device)

        make_dirs(ckpt)

        best_weights = copy.deepcopy(self.model.state_dict())

        best_loss = float("inf")
        correct = 0.0
        total = 0.0
        running_loss = 0.0

        for epoch in range(epochs):
            for images, labels in tqdm(dataloader["train"]):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()

                optimizer.step()

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                if loss < best_loss:
                    best_loss = loss
                    best_weights = copy.deepcopy(self.model.state_dict())

            train_loss = running_loss / len(dataset["train"])

            logging.info(
                f"Epoch: [{epoch+1}/{epochs}],Train Acc: {correct/total:.3f} ,Loss: {train_loss:.3f}"
            )

            test_loss, test_accuracy = self.evaluate(
                self.model, criterion, dataset, dataloader
            )

            logging.info(
                f"Epoch: [{epoch+1}/{epochs}],Test Acc: {test_accuracy:.3f} ,Loss: {test_loss:.3f}"
            )

            if epoch % self.save_period == 0 and epoch:
                torch.save(self.model.state_dict(), ckpt + f"/checkpoint-{epoch}.pt")

        self.model.load_state_dict(best_weights)

        test_loss, test_accuracy = self.evaluate(
            self.model, criterion, dataset, dataloader
        )
        logging.info(f"Final Acc: {test_accuracy:.3f} ,Loss: {test_loss:.3f}")
        torch.save(self.model.state_dict(), ckpt + "/best.pt")

    def evaluate(self, model, criterion, dataset, dataloader):
        correct = 0.0
        total = 0.0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(dataloader["test"]):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                running_loss += loss.item()
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        total_loss = running_loss / len(dataset["test"])
        return total_loss, correct / total
