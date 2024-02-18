import shutil
from pathlib import Path
import uuid
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="../data/dataset", type=str)
    parser.add_argument("-i", "--input", default="../data/train", type=str)
    parser.add_argument("--sample", default=50, type=int)
    return parser.parse_args()


def make_dirs(args):
    Path(args["output"]).mkdir(parents=True, exist_ok=True)


def merge_dirs(input_dir, output_dir, sample_size):
    for dir_path in tqdm(list(Path(input_dir).iterdir())):
        for fpath in list(dir_path.iterdir())[:sample_size]:
            shutil.copy(fpath, "../data/dataset/" + str(uuid.uuid4()) + ".jpg")


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    logging.info(args)
    make_dirs(args)
    merge_dirs(args["input"], args["output"], args["sample"])
