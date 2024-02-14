from torch.utils.data import Dataset
from pinaka.dataset import ImageDataset

def test_dataset():
    data_path = "./data/dataset/"
    dataset = ImageDataset(data_path)
    assert isinstance(dataset,Dataset) 
