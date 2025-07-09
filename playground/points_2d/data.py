import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_swiss_roll


def make_swiss_roll_dataframe(
    n_samples: int, n_dims: int = 2, noise: float = 0.0, scaling_factor: float = 10.0
) -> pd.DataFrame:
    coords_3d, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    coords_3d /= scaling_factor
    if n_dims == 2:
        return pd.DataFrame({"x": coords_3d[:, 0], "y": coords_3d[:, 2]})
    else:
        assert n_dims == 3
        return pd.DataFrame({"x": coords_3d[:, 0], "y": coords_3d[:, 1], "z": coords_3d[:, 2]})


class SwissRollDataset(torch.utils.data.Dataset):
    """Custom dataset for Swiss roll data that returns dict with 'input' key."""
    
    def __init__(self, n_samples: int, n_dims: int = 2, noise: float = 0.0):
        self.n_dims = n_dims
        df = make_swiss_roll_dataframe(n_samples=n_samples, n_dims=n_dims, noise=noise)
        self.data = torch.from_numpy(df.to_numpy().astype(np.float32))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"input": self.data[idx]}


def create_train_val_datasets(
    n_train: int = 2**12, n_val: int = 2**9, n_dims: int = 2, noise: float = 0.0
):
    train_dataset = SwissRollDataset(n_samples=n_train, n_dims=n_dims, noise=noise)
    val_dataset = SwissRollDataset(n_samples=n_val, n_dims=n_dims, noise=noise)
    return train_dataset, val_dataset
