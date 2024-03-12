from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class RawDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_paths = list(Path(self.root_dir).iterdir())

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.transforms:
            img = self.transforms(img)
        return img
