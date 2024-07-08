import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class Clusterdata(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = dataframe.values

    def len(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        x = torch.tensor(self.features(index), dtype=torch.float32)
        return x


def load_data(file_path, batch_size, shuffle=True):
    df = pd.read_csv(file_path)
    dataset = Clusterdata(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
