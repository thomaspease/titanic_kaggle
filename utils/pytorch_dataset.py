from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from utils.data_loader import TitanicLoader

class TitanicPytorchDataset(Dataset):
  def __init__(self, label):
    data = TitanicLoader()
    self.X, self.y = data.load(label)

  def __getitem__(self, index):
    # The classes have to be cast as long
    features = torch.tensor(self.X.iloc[index], dtype=torch.float32)
    label = torch.tensor(self.y.iloc[index], dtype=torch.long)

    return (features, label)

  def __len__(self):
    return len(self.X)

dataset = TitanicPytorchDataset('Survived')

split_datasets = {}
split_datasets['train'], split_datasets['val'] = random_split(dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

data_loaders = {}
data_loaders['train'] = DataLoader(split_datasets['train'], batch_size=8, shuffle=True)
data_loaders['val'] = DataLoader(split_datasets['val'], batch_size=8, shuffle=True)