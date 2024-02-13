import torch
from imgcluster.models import *


def test_model():
    autoencoder = AutoEncoder()
    x = torch.randn(1, 3, 224, 224)
    y = autoencoder(x)
    assert x.size() == y.size()
