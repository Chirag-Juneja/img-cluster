import shutil
from pathlib import Path
import uuid
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="../data/subtrain", type=str)
    parser.add_argument("-i", "--input", default="../data/train", type=str)
    parser.add_argument("--sample", default=50, type=int)
    return parser.parse_args()


def make_subset(args):
    train_path = Path(args["input"])
    subtrain_path = Path(args["output"])
    sample_size = args["sample"]

    for dir_path in tqdm(list(train_path.iterdir())):
        cls_path = subtrain_path.joinpath(dir_path.parts[-1])
        cls_path.mkdir(parents=True, exist_ok=True)
        for fpath in list(dir_path.iterdir())[:sample_size]:
            shutil.copy(fpath, str(cls_path) + "/" + str(uuid.uuid4()) + ".jpg")


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    logging.info(args)
    make_subset(args)
