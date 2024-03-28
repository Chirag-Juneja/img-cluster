import json
from pathlib import Path


def load_json(fpath: str):
    with open(fpath) as f:
        data = json.load(f)
    return data


def make_dirs(dirs, **kwargs):
    Path(dirs).mkdir(parents=True, exist_ok=True)
