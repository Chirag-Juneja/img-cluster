import pytest
import torch
from pinaka.models.effb0cl import EffB0CL


class TestEffB0CL:
    classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def test_output_dim(self):
        model = EffB0CL(self.classes)
        x = torch.randn(4, 3, 128, 128)
        y = model(x)
        assert y.size()[1] == len(self.classes)
