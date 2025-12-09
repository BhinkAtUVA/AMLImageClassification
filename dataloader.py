import pandas as pd
import torch as tc
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

from torchvision.transforms import v2

TRANSFORM_DEFAULT = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(tc.float32, scale=True),
    ]
)

DATA_DIR = Path("./data")

# idx, image, label, path
ItemType = tuple[int, tc.Tensor, int, str]

class BirdData(Dataset[ItemType]):
    # Inputs
    data_dir: Path
    filename: str

    # Attributes
    df: pd.DataFrame

    def __init__(
        self,
        filename: str,
        data_dir: Path = DATA_DIR,
        transform: v2.Transform = TRANSFORM_DEFAULT,
    ):
        self.data_dir = data_dir
        self.filename = filename
        self.transform = transform
        self.filepath = self.data_dir / self.filename

        if not self.filepath.exists():
            raise FileNotFoundError(f"DB file not found")

        df = pd.read_csv(self.filepath)

        assert isinstance(df, pd.DataFrame)
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> ItemType:
        row = self.df.iloc[idx]

        idx = int(row.id or idx)
        label = int(row.label)
        path = str(self.data_dir / row.image_path)

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return idx, image, label, path