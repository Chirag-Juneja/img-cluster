import torch
import logging
from pathlib import Path
from .models.effb0ae import EffB0AE


class Classificaiton:
    def __init__(self, model=None):

        path = Path(model)
        self.model = EffB0AE()

        if path.is_file():
            try:
                model.load_state_dict(torch.load(path))
                logging.info(f"Model weights restored from {path}")
            except Exception as e:
                logging.warning(f"Weights load failed. {path}\n {e}")
