import os
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.len = self.get_len()

    def get_len(self):
        len = 0
        for _ in os.scandir(self.img_dir):
            len += 1
        print(len)
        return len

    def __len__(self):
        return self.len