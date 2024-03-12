import json


def load_json(fpath: str):
    with open(fpath) as f:
        data = json.load(f)
    return data
