import shutil
from pathlib import Path
import uuid
from tqdm import tqdm


for dir_path in tqdm(list(Path("../data/train/").iterdir())):
    for fpath in dir_path.iterdir():
        shutil.copy(fpath, "../data/dataset/"+str(uuid.uuid4())+".jpg")
