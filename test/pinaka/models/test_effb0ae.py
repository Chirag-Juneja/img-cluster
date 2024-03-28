import torch
from pinaka.models.effb0ae import EffB0AE


class TestEffB0AE:
    def test_output_size(self):
        model = EffB0AE()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert x.size() == y.size()
