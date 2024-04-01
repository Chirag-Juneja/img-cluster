import argparse
import logging
import multiprocessing
from pinaka.utils.io import load_json
from pinaka.classification import Classificaiton

logging.basicConfig(level=logging.INFO)
NUM_WORKER = multiprocessing.cpu_count() // 2


def parse_args():
    parser = argparse.ArgumentParser("PreTrian Task Model Training ")
    parser.add_argument("-b", "--batch", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--imgsz", default=32, type=int)
    parser.add_argument("--ckpt", default="./checkpoints", type=str)
    parser.add_argument("--save_period", default=10, type=str)
    parser.add_argument("--data", default="./dataset", type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--config", default=None, type=str)
    return parser.parse_args()


def pretrain(
    epochs=100,
    batch=64,
    device="cuda",
    data="./dataset",
    imgsz=96,
    ckpt="./checkpoints/",
    weights=None,
    save_period=10,
    classes="classes.txt",
    train_path="./data/test",
    test_path="./data/val",
    **kwargs,
):
    with open(classes) as f:
        lines = f.readlines()
    classes = [c[:-1] for c in lines]

    # classification = Classificaiton(device=device, save_period=save_period,classes=classes)
    # classification.train(train_path, test_path, epochs=epochs, batch=1024, ckpt=ckpt+"cls/", imgsz=imgsz)

    classification = Classificaiton(
        effb0ae="checkpoints/pretrain/best.pt",
        device=device,
        save_period=save_period,
        classes=classes,
    )
    # classification.pretrain(data, epochs=100, batch=1024, ckpt=ckpt+"pretrain/", imgsz=imgsz)
    classification.train(
        train_path,
        test_path,
        epochs=epochs,
        batch=1024,
        ckpt=ckpt + "pretrain_cls/",
        imgsz=imgsz,
        features=classification.effb0ae.encoder,
    )


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    if args["config"]:
        args = load_json(args["config"])
    logging.info(args)
    pretrain(**args)
