from torch.utils.data import Dataset
from pinaka.datasets import RawDataset


class TestRawDataset:

    def test_instance(self):
        data_path = "./data/dataset/"
        dataset = RawDataset(data_path)
        assert isinstance(dataset, Dataset)
