import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):

    def __init__(self, df, transform=None):
        '''init dataset'''
        self.df = df
        self.transform = transform

    def __len__(self):
        '''init return length of dataset'''
        return self.df.shape[0]

    def __getitem__(self, index):
        '''define label and transform image based on index given'''
        label = self.df.iloc[index, 0]
        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label
